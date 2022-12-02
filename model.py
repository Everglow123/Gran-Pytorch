"""
@文件    :model.py
@时间    :2022/11/17 00:06:29
@作者    :周恒
@版本    :1.0
@说明    :
"""


from ast import Not
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    Dropout,
    Embedding,
    LayerNorm,
    Module,
    Sequential,
    Linear,
)
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from data_process import NaryFeature, Vocabulary, generate_ground_truth


@dataclass
class GranConfig:
    voc_size: int
    n_relation: int
    n_edge: int
    max_seq_len: int
    max_arity: int
    n_layer: int = 12
    n_head: int = 4
    emb_size: int = 256
    intermediate_size: int = 512
    hidden_act: str = "gelu"
    prepostprocess_dropout: float = 0.1
    attention_dropout: float = 0.1
    initializer_range: float = 0.02
    e_soft_label: float = 1.0
    r_soft_label: float = 1.0


def gen_act_func(act: str) -> Callable[[Tensor], Tensor]:
    act = act.lower()
    res = eval("F." + act)
    return res


class GranPrePostProcessLayer(Module):
    def __init__(
        self,
        shape: Sequence[int],
        process_cmd: str,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        layers: List[Module] = []
        for cmd in process_cmd:
            if cmd == "n":
                layer = LayerNorm(shape, dtype=torch.float32)
                layers.append(layer)
            elif cmd == "d":
                rate = 0.5
                if dropout_rate:
                    rate = dropout_rate
                    layer = Dropout(rate)
                    layers.append(layer)
        self.layers: Sequential = Sequential(*layers)

    def forward(self, input: Tensor, prev=None):
        if prev is not None:
            input = prev + input
        return self.layers(input)


class GranPositionWiseFeedForward(Module):
    def __init__(
        self,
        emb_size: int,
        d_inner_hid: int,
        dropout_rate: float,
        hidden_act=Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.d_inner_hid = d_inner_hid
        self.fc1 = Linear(emb_size, d_inner_hid)
        self.act = hidden_act
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(d_inner_hid, emb_size)

    def forward(self, x: Tensor):
        hidden = self.fc1(x)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        return out


class GranMultiHeadAttention(Module):
    def __init__(
        self,
        emb_size: int,
        d_key: int,
        d_value: int,
        d_model: int,
        n_head: int = 1,
        dropout_rate=0.0,
    ) -> None:
        super().__init__()

        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.query_fc = Linear(emb_size, self.d_key * self.n_head)
        self.key_fc = Linear(emb_size, self.d_key * self.n_head)
        self.value_fc = Linear(emb_size, self.d_key * self.n_head)
        self.proj_fc = Linear(emb_size, self.d_model)

    def compute_qkv(
        self, queries: Tensor, keys: Tensor, values: Tensor
    ) -> Tuple[Tensor, ...]:
        # queries: batch_size*max_seq_len*emb_size
        # q: batch_size*max_seq_len*(d_key*self.n_head)
        q = self.query_fc(queries)
        k = self.key_fc(keys)
        v = self.value_fc(values)
        return q, k, v

    def split_heads(self, x: Tensor):
        hidden_size = x.shape[-1]
        reshaped = torch.reshape(
            x, shape=[x.shape[0], x.shape[1], self.n_head, hidden_size // self.n_head]
        )
        res = torch.transpose(reshaped, 1, 2)
        return res

    def edge_aware_self_attention(
        self,
        q: Tensor,  # [B, N, M, H]
        k: Tensor,  # [B, N, M, H]
        v: Tensor,  # [B, N, M, H]
        edges_k: Tensor,  # [M, M, H]
        edges_v: Tensor,  # [M, M, H]
        attn_bias: Tensor,  # [B, N, M, M]
    ):
        if not (len(q.shape) == len(k.shape) == len(v.shape) == 4):
            raise ValueError("Input q, k, v should be 4-D Tensors.")
        if not (len(edges_k.shape) == len(edges_v.shape) == 3):
            raise ValueError("Input edges_k and edges_v should be 3-D Tensors.")
        scaled_q = q * (self.d_key**-0.5)
        product = torch.matmul(scaled_q, k.transpose(2, 3))
        if edges_k is not None and edges_v is not None:
            scaled_q = scaled_q.permute([2, 0, 1, 3])
            scaled_q = torch.reshape(scaled_q, shape=[scaled_q.shape[0], -1, scaled_q.shape[3]])
            edge_bias = torch.matmul(scaled_q, edges_k.transpose(1, 2))
            edge_bias = torch.reshape(
                edge_bias,
                [edge_bias.shape[0], -1, q.shape[1], q.shape[2]],
            )
            edge_bias = torch.permute(edge_bias, [1, 2, 0, 3])
            product += edge_bias
            product += attn_bias
        weights = F.softmax(product)
        if self.dropout_rate:
            weights = F.dropout(weights, self.dropout_rate)
        out = torch.matmul(weights, v)
        
        reshaped_weights = weights.permute([2, 0, 1, 3])
        reshaped_weights = reshaped_weights.reshape(
            [reshaped_weights.shape[0], -1, reshaped_weights.shape[3]]
        )
        edge_bias = torch.matmul(reshaped_weights, edges_v)
        edge_bias = edge_bias.reshape(
            shape=[edge_bias.shape[0], -1, q.shape[1], q.shape[3]]
        )
        edge_bias = edge_bias.permute([1, 2, 0, 3])
        out += edge_bias
        return out

    def combine_heads(self, x: Tensor):
        if len(x.shape) == 3:
            return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        trans_x = x.permute([0, 2, 1, 3])
        trans_x = trans_x.reshape([trans_x.shape[0], trans_x.shape[1], -1])
        return trans_x

    def forward(
        self,
        queries: Optional[Tensor],
        keys: Optional[Tensor],
        values: Optional[Tensor],
        edges_key: Tensor,
        edges_value: Tensor,
        attn_bias: Tensor,
        cache: Optional[Dict[str, Tensor]] = None,
    ):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors."
            )
        q, k, v = self.compute_qkv(queries, keys, values)

        if cache is not None:
            old_shape = cache["k"].shape
            k = cache["k"] = torch.cat(
                [
                    torch.reshape(
                        cache["k"], [old_shape[0], old_shape[1], self.d_model]
                    ),
                    k,
                ],
                dim=1,
            )
            old_shape = cache["v"].shape
            v = cache["v"] = torch.cat(
                [
                    torch.reshape(
                        cache["v"], [old_shape[0], old_shape[1], self.d_model]
                    ),
                    v,
                ],
                dim=1,
            )
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        ctx_multiheads = self.edge_aware_self_attention(
            q, k, v, edges_key, edges_value, attn_bias
        )
        out = self.combine_heads(ctx_multiheads)
        proj_out = self.proj_fc(out)
        return proj_out


class GranEncoderLayer(Module):
    def __init__(
        self,
        max_seq_len: int,
        emb_size: int,
        intermediate_size: int,
        d_key: int,
        d_value: int,
        d_model: int,
        n_head: int,
        prepostprocess_dropout: float,
        attention_dropout: float,
        relu_dropout: float,
        hidden_act: Callable[[Tensor], Tensor],
        preprocess_cmd="",
        postprocess_cmd="dan",
    ) -> None:
        super().__init__()
        self.post_process_layer1 = GranPrePostProcessLayer(
            [max_seq_len, emb_size], preprocess_cmd, prepostprocess_dropout
        )
        self.multi_head_attention = GranMultiHeadAttention(
            emb_size, d_key, d_value, d_model, n_head, attention_dropout
        )
        self.post_process_layer2 = GranPrePostProcessLayer(
            [max_seq_len, emb_size], postprocess_cmd, prepostprocess_dropout
        )
        self.positionwise_feed_forward = GranPositionWiseFeedForward(
            emb_size, intermediate_size, relu_dropout, hidden_act
        )
        self.post_process_layer3 = GranPrePostProcessLayer(
            [max_seq_len, emb_size], postprocess_cmd, prepostprocess_dropout
        )

    def forward(
        self,
        enc_input: Tensor,
        edges_key: Tensor,
        edges_value: Tensor,
        atten_bias: Tensor,
    ):
        enc_input = self.post_process_layer1(enc_input)
        attn_output: Tensor = self.multi_head_attention(
            enc_input, None, None, edges_key, edges_value, atten_bias
        )
        attn_output = self.post_process_layer2(attn_output, enc_input)
        ffd_output = self.positionwise_feed_forward(attn_output)
        res = self.post_process_layer3(ffd_output, attn_output)
        return res


class GranEncoder(Module):
    def __init__(
        self,
        max_seq_len: int,
        emb_size: int,
        n_layer: int,
        d_key: int,
        d_value: int,
        d_model: int,
        n_head: int,
        d_inner_hid: int,
        prepostprocess_dropout: float,
        attention_dropout: float,
        relu_dropout: float,
        hidden_act=F.relu,
        preprocess_cmd="",
        postprocess_cmd="dan",
    ) -> None:
        super().__init__()
        self.n_layer = n_layer
        layers = []
        for i in range(n_layer):
            layers.append(
                GranEncoderLayer(
                    max_seq_len,
                    emb_size,
                    d_inner_hid,
                    d_key,
                    d_value,
                    d_model,
                    n_head,
                    prepostprocess_dropout,
                    attention_dropout,
                    relu_dropout,
                    hidden_act,
                    preprocess_cmd,
                    postprocess_cmd,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.pre_post_process_layer = GranPrePostProcessLayer(
            [max_seq_len, emb_size], preprocess_cmd, prepostprocess_dropout
        )

    def forward(
        self,
        enc_input: Tensor,
        edges_key: Tensor,
        edges_value: Tensor,
        attn_bias: Tensor,
    ):
        for i in range(self.n_layer):
            layer: GranEncoderLayer = self.layers[i]
            enc_output = layer(enc_input, edges_key, edges_value, attn_bias)
            enc_input = enc_output
        enc_output = self.pre_post_process_layer(enc_input)
        return enc_output


class GranModel(Module):
    def __init__(
        self,config: GranConfig, weight_sharing=True
    ) -> None:
        super().__init__()
        self.weight_sharing = weight_sharing
        self.config = config
        self.pre_post_process_layer1 = GranPrePostProcessLayer(
            [self.config.max_seq_len, self.config.emb_size],
            "nd",
            self.config.prepostprocess_dropout,
        )
        self.ent_embedding = Embedding(self.config.voc_size, self.config.emb_size)
        self.edge_key_embedding = Embedding(
            self.config.n_edge, self.config.emb_size // self.config.n_head
        )
        self.edge_value_embedding = Embedding(
            self.config.n_edge, self.config.emb_size // self.config.n_head
        )
        self.encoder = GranEncoder(
        
            max_seq_len=config.max_seq_len,
            emb_size=config.emb_size,
            n_layer=config.n_layer,
            d_key=config.emb_size // config.n_head,
            d_value=config.emb_size // config.n_head,
            d_model=config.emb_size,
            n_head=config.n_head,
            d_inner_hid=config.intermediate_size,
            prepostprocess_dropout=config.prepostprocess_dropout,
            attention_dropout=config.attention_dropout,
            relu_dropout=0.0,
            hidden_act=gen_act_func(config.hidden_act),
            preprocess_cmd="",
            postprocess_cmd="dan",
        )

        self.mask_trans_feat_fc = Linear(self.config.emb_size, self.config.emb_size)
        self.mask_trans_feat_act_func = gen_act_func(self.config.hidden_act)
        self.mask_trans_feat_pre_process_layer = GranPrePostProcessLayer(
            shape=[config.emb_size], process_cmd="n"
        )
        self.mask_lm_out_bias_attr = torch.nn.parameter.Parameter(
            data=torch.zeros([self.config.voc_size])
        )
        self.mask_lm_out_fc = Linear(self.config.emb_size, self.config.voc_size)
        
    def _build_edge_labels(self) -> torch.Tensor:
        edge_labels = []
        max_aux = self.config.max_arity - 2
        edge_labels.append([0, 1, 2] + [3] * max_aux + [0] * max_aux)
        edge_labels.append([1] + [0] * (self.config.max_seq_len - 1))
        edge_labels.append([2] + [0] * (self.config.max_seq_len  - 1))
        for idx in range(max_aux):
            edge_labels.append(
                [3, 0, 0] + [0] * max_aux + [0] * idx + [4] + [0] * (max_aux - idx - 1)
            )
        for idx in range(max_aux):
            edge_labels.append(
                [0, 0, 0] + [0] * idx + [4] + [0] * (max_aux - idx - 1) + [0] * max_aux
            )
        edge_labels = (
            np.asarray(edge_labels)
            .astype("int64")
            .reshape([self.config.max_seq_len , self.config.max_seq_len ])
        )
        edge_labels = torch.from_numpy(edge_labels)
        return edge_labels
    def forward(
        self,
        input_ids: Tensor,  # batch_size * max_seq_len
        input_mask: Tensor,
        mask_pos: Tensor,  # [batch_size],
        mask_type: Tensor,  # batch_size
    ):

        edge_labels: Tensor=self._build_edge_labels().to(device=input_ids.device)  # max_seq_len * max_seq_len * 1,

        entity_emb = self.ent_embedding(input_ids)
        """batch_size*max_seq_size*hidden_size"""
        emb_out = self.pre_post_process_layer1(entity_emb)
        """(max_seq_len * max_seq_len)"""
        edge_labels = edge_labels.reshape([-1])

        """(max_seq_len * max_seq_len) * (config.emb_size//self.config.n_head)"""
        edges_key: Tensor = self.edge_key_embedding(edge_labels)

        """(max_seq_len * max_seq_len) * (config.emb_size//self.config.n_head)"""
        edges_value: Tensor = self.edge_value_embedding(edge_labels)

        """(max_seq_len * max_seq_len) """
        edge_mask = torch.sign(edge_labels.float()).unsqueeze(-1)
        edges_key = edges_key * edge_mask
        edges_value = edges_value * edge_mask
        edges_key=edges_key.reshape( [self.config.max_seq_len, self.config.max_seq_len, -1])
        edges_value = edges_value.reshape(
            [self.config.max_seq_len, self.config.max_seq_len, -1]
        )
        attn_mask = torch.matmul(input_mask, input_mask.transpose(1, 2))
        attn_mask = 1000000.0 * (attn_mask - 0.1)
        n_head_self_attn_mask = [attn_mask] * self.config.n_head
        n_head_self_attn_mask = torch.stack(n_head_self_attn_mask, dim=1)
        n_head_self_attn_mask = n_head_self_attn_mask.detach_()

        # batch_size * max_seq_len * emb_size
        enc_out = self.encoder(emb_out, edges_key, edges_value, n_head_self_attn_mask)
        batch_size = mask_pos.shape[0]
        # batch_size * emb_size
        mask_feat = enc_out[
            torch.arange(batch_size, device=mask_pos.device), mask_pos, :
        ]
        mask_trans_feat = self.mask_trans_feat_fc(mask_feat)
        mask_trans_feat = self.mask_trans_feat_act_func(mask_trans_feat)
        mask_trans_feat = self.mask_trans_feat_pre_process_layer(mask_trans_feat)
        fc_out: Tensor = None
        if self.weight_sharing:
            # [batch_size * emb_size] matmul [emb_size * vocab_size]
            # batch_size * vocab_size
            fc_out = torch.matmul(
                mask_trans_feat, self.ent_embedding.weight.transpose(0, 1)
            )
            fc_out += self.mask_lm_out_bias_attr
        else:
            fc_out = self.mask_lm_out_fc(mask_trans_feat)

        special_indicator = torch.full(
            [batch_size, 2], fill_value=-1, dtype=torch.long, device=fc_out.device
        )
        relation_indicator = torch.full(
            [batch_size, self.config.n_relation],
            fill_value=-1,
            dtype=torch.long,
            device=fc_out.device,
        )
        entity_indicator = torch.full(
            [batch_size, self.config.voc_size - self.config.n_relation - 2],
            fill_value=1,
            dtype=torch.long,
            device=fc_out.device,
        )
        # batch_size * (voc_size-2)
        type_indicator = torch.cat([relation_indicator, entity_indicator], dim=-1)
        mask_type = mask_type.reshape([-1, 1])
        type_indicator = type_indicator * mask_type
        type_indicator = torch.cat([special_indicator, type_indicator], dim=-1)
        type_indicator = type_indicator.to(dtype=torch.float32)
        type_indicator = F.relu(type_indicator)

        fc_out_mask = (type_indicator - 1.0) * 1000000.0
        fc_out = fc_out + fc_out_mask
        # batch_size * vocab_size
        return fc_out, type_indicator

def get_optimizer(model:GranModel,lr:float):
    optimizer=torch.optim.Adam([
        {"params":model.parameters(),"lr":lr}
    ],lr=lr)
    return optimizer

def batch_cal_loss_func(
    labels: Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...], trainer
) -> Tensor:
    model: GranModel = trainer.model
    if isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module

    mask_labels, mask_type, _ = labels
    fc_out, type_indicator = preds
    mask_labels = mask_labels.reshape([-1])
    batch_size = mask_labels.shape[0]
    # batch_size * vocab_size
    one_hot_labels: Tensor = F.one_hot(mask_labels, model.config.voc_size)
    type_indicator = type_indicator - one_hot_labels

    # batch_size
    num_candidates = type_indicator.sum(dim=-1)

    mask_type = mask_type.reshape([-1, 1])
    mask_type = mask_type.to(dtype=torch.float32)

    soft_labels = (
        (1.0 + mask_type) * model.config.e_soft_label
        + (1.0 - mask_type) * model.config.r_soft_label
    ) / 2.0
    # batch_size*voc_size
    soft_labels = soft_labels.repeat([1, model.config.voc_size])
    soft_labels = soft_labels * one_hot_labels + (1.0 - soft_labels) * (
        type_indicator / num_candidates.unsqueeze(-1)
    )

    log_out = F.log_softmax(fc_out, dim=1)
    loss = (-(log_out * soft_labels)).sum(-1) / batch_size
    return loss.mean()


def batch_forward_func(batch_data: Tuple[torch.Tensor, ...], trainer):
    batch_data_to_dev: List[Tensor] = []
    for item in batch_data:
        if isinstance(item, Tensor):
            batch_data_to_dev.append(item.to(device=trainer.device, non_blocking=True))
        else:
            batch_data_to_dev.append(item)
    (
        batch_input_ids,
        batch_input_mask,
        batch_mask_position,
        batch_mask_label,
        batch_mask_type, 
        batch_features,
    ) = batch_data_to_dev
    fc_out, type_indicator = trainer.model(
        batch_input_ids,
        batch_input_mask, 
        batch_mask_position,
        batch_mask_type,
    )
    return (batch_mask_label, batch_mask_type, batch_features), (fc_out, type_indicator)


class BatchMetricsFunc:
    def __init__(
        self,
        ground_truth_path: str,
        vocabulary: Vocabulary,
        max_arity: int,
        max_seq_length: int,
    ) -> None:
        self.ground_truth_path = ground_truth_path
        self.vocabulary = vocabulary
        self.max_arity = max_arity
        self.max_seq_length = max_seq_length
        self.gt_dict = generate_ground_truth(
            ground_truth_path=ground_truth_path,
            vocabulary=vocabulary,
            max_arity=max_arity,
            max_seq_length=max_seq_length,
        )

    def __call__(
        self,
        labels: Tuple[torch.Tensor, torch.Tensor, List[NaryFeature]],
        preds: Tuple[torch.Tensor, ...],
        metrics: Dict[str, float],
        trainer,
    ) -> Any:
        model: GranModel = trainer.model
        if isinstance(model, torch.nn.parallel.DataParallel):
            model = model.module
        mask_labels, mask_type, batch_features = labels
        fc_out, _ = preds
        fc_out=fc_out.cpu().detach()
        ret_ranks = {
            "entity": [],
            "relation": [],
            "2-r": [],
            "2-ht": [],
            "n-r": [],
            "n-ht": [],
            "n-a": [],
            "n-v": [],
        }
        batch_size = len(batch_features)
        for i, feature in enumerate(batch_features):
            result = fc_out[i].numpy()
            target = feature.mask_label
            pos = feature.mask_position
            key = " ".join(
                [
                    str(feature.input_ids[x])
                    for x in range(len(feature.input_ids))
                    if x != pos
                ]
            )
            rm_idx = self.gt_dict[pos][key]
            rm_idx = [x for x in rm_idx if x != target]
            for x in rm_idx:
                result[x] = -np.Inf
            sortidx = np.argsort(result)[::-1]

            if feature.mask_type == 1:
                ret_ranks["entity"].append(np.where(sortidx == target)[0][0] + 1)
            elif feature.mask_type == -1:
                ret_ranks["relation"].append(np.where(sortidx == target)[0][0] + 1)
            else:
                raise ValueError("Invalid `feature.mask_type`.")

            if feature.arity == 2:
                if pos == 0:
                    ret_ranks["2-r"].append(np.where(sortidx == target)[0][0] + 1)
                elif pos == 1 or pos == 2:
                    ret_ranks["2-ht"].append(np.where(sortidx == target)[0][0] + 1)
                else:
                    raise ValueError("Invalid `feature.mask_position`.")
            elif feature.arity > 2:
                if pos == 0:
                    ret_ranks["n-r"].append(np.where(sortidx == target)[0][0] + 1)
                elif pos == 1 or pos == 2:
                    ret_ranks["n-ht"].append(np.where(sortidx == target)[0][0] + 1)
                elif pos > 2 and feature.mask_type == -1:
                    ret_ranks["n-a"].append(np.where(sortidx == target)[0][0] + 1)
                elif pos > 2 and feature.mask_type == 1:
                    ret_ranks["n-v"].append(np.where(sortidx == target)[0][0] + 1)
                else:
                    raise ValueError("Invalid `feature.mask_position`.")
            else:
                raise ValueError("Invalid `feature.arity`.")
        ent_ranks = np.asarray(ret_ranks["entity"])
        rel_ranks = np.asarray(ret_ranks["relation"])
        _2_r_ranks = np.asarray(ret_ranks["2-r"])
        _2_ht_ranks = np.asarray(ret_ranks["2-ht"])
        _n_r_ranks = np.asarray(ret_ranks["n-r"])
        _n_ht_ranks = np.asarray(ret_ranks["n-ht"])
        _n_a_ranks = np.asarray(ret_ranks["n-a"])
        _n_v_ranks = np.asarray(ret_ranks["n-v"])
        eval_result={}
        temp=[ 
            ent_ranks,
            rel_ranks,
            _2_r_ranks,
            _2_ht_ranks,
            _n_r_ranks,
            _n_ht_ranks,
            _n_a_ranks,
            _n_v_ranks
        ]
        
        for item in [
            "ent_lst",
            "rel_lst",
            "_2_r_lst",
            "_2_ht_lst",
            "_n_r_lst",
            "_n_ht_lst",
            "_n_a_lst",
            "_n_v_lst",
        ]:
            if item not in metrics:
                metrics[item] = []
        ent_lst = metrics["ent_lst"]
        rel_lst = metrics["rel_lst"]
        _2_r_lst = metrics["_2_r_lst"]
        _2_ht_lst = metrics["_2_ht_lst"]
        _n_r_lst = metrics["_n_r_lst"]
        _n_ht_lst = metrics["_n_ht_lst"]
        _n_a_lst = metrics["_n_a_lst"]
        _n_v_lst = metrics["_n_v_lst"]
        ent_lst.extend(ent_ranks)
        rel_lst.extend(rel_ranks)
        _2_r_lst.extend(_2_r_ranks)
        _2_ht_lst.extend(_2_ht_ranks)
        _n_r_lst.extend(_n_r_ranks)
        _n_ht_lst.extend(_n_ht_ranks)
        _n_a_lst.extend(_n_a_ranks)
        _n_v_lst.extend(_n_v_ranks)

        return metrics,ret_ranks


class MetricsCalFunc:
    def __init__(self) -> None:
        pass

    def __call__(self, metrics: Dict[str, np.ndarray]) -> Any:
        ent_lst = metrics["ent_lst"]
        rel_lst = metrics["rel_lst"]
        _2_r_lst = metrics["_2_r_lst"]
        _2_ht_lst = metrics["_2_ht_lst"]
        _n_r_lst = metrics["_n_r_lst"]
        _n_ht_lst = metrics["_n_ht_lst"]
        _n_a_lst = metrics["_n_a_lst"]
        _n_v_lst = metrics["_n_v_lst"]
        eval_result = compute_metrics(
            ent_lst,
            rel_lst,
            _2_r_lst,
            _2_ht_lst,
            _n_r_lst,
            _n_ht_lst,
            _n_a_lst,
            _n_v_lst,
        )
        return eval_result

def compute_metrics(
    ent_lst, rel_lst, _2_r_lst, _2_ht_lst, _n_r_lst, _n_ht_lst, _n_a_lst, _n_v_lst
):
    """
    Combine the ranks from batches into final metrics.
    """
    all_ent_ranks = np.array(ent_lst).ravel()
    all_rel_ranks = np.array(rel_lst).ravel()
    _2_r_ranks = np.array(_2_r_lst).ravel()
    _2_ht_ranks = np.array(_2_ht_lst).ravel()
    _n_r_ranks = np.array(_n_r_lst).ravel()
    _n_ht_ranks = np.array(_n_ht_lst).ravel()
    _n_a_ranks = np.array(_n_a_lst).ravel()
    _n_v_ranks = np.array(_n_v_lst).ravel()
    all_r_ranks = np.array(_2_r_lst + _n_r_lst).ravel()
    all_ht_ranks = np.array(_2_ht_lst + _n_ht_lst).ravel()
    mrr_ent = np.mean(1.0 / all_ent_ranks)
    hits1_ent = np.mean(all_ent_ranks <= 1.0)
    hits3_ent = np.mean(all_ent_ranks <= 3.0)
    hits5_ent = np.mean(all_ent_ranks <= 5.0)
    hits10_ent = np.mean(all_ent_ranks <= 10.0)
    mrr_rel = np.mean(1.0 / all_rel_ranks)
    hits1_rel = np.mean(all_rel_ranks <= 1.0)
    hits3_rel = np.mean(all_rel_ranks <= 3.0)
    hits5_rel = np.mean(all_rel_ranks <= 5.0)
    hits10_rel = np.mean(all_rel_ranks <= 10.0)
    mrr_2r = np.mean(1.0 / _2_r_ranks)
    hits1_2r = np.mean(_2_r_ranks <= 1.0)
    hits3_2r = np.mean(_2_r_ranks <= 3.0)
    hits5_2r = np.mean(_2_r_ranks <= 5.0)
    hits10_2r = np.mean(_2_r_ranks <= 10.0)
    mrr_2ht = np.mean(1.0 / _2_ht_ranks)
    hits1_2ht = np.mean(_2_ht_ranks <= 1.0)
    hits3_2ht = np.mean(_2_ht_ranks <= 3.0)
    hits5_2ht = np.mean(_2_ht_ranks <= 5.0)
    hits10_2ht = np.mean(_2_ht_ranks <= 10.0)
    mrr_nr = np.mean(1.0 / _n_r_ranks)
    hits1_nr = np.mean(_n_r_ranks <= 1.0)
    hits3_nr = np.mean(_n_r_ranks <= 3.0)
    hits5_nr = np.mean(_n_r_ranks <= 5.0)
    hits10_nr = np.mean(_n_r_ranks <= 10.0)
    mrr_nht = np.mean(1.0 / _n_ht_ranks)
    hits1_nht = np.mean(_n_ht_ranks <= 1.0)
    hits3_nht = np.mean(_n_ht_ranks <= 3.0)
    hits5_nht = np.mean(_n_ht_ranks <= 5.0)
    hits10_nht = np.mean(_n_ht_ranks <= 10.0)
    mrr_na = np.mean(1.0 / _n_a_ranks)
    hits1_na = np.mean(_n_a_ranks <= 1.0)
    hits3_na = np.mean(_n_a_ranks <= 3.0)
    hits5_na = np.mean(_n_a_ranks <= 5.0)
    hits10_na = np.mean(_n_a_ranks <= 10.0)
    mrr_nv = np.mean(1.0 / _n_v_ranks)
    hits1_nv = np.mean(_n_v_ranks <= 1.0)
    hits3_nv = np.mean(_n_v_ranks <= 3.0)
    hits5_nv = np.mean(_n_v_ranks <= 5.0)
    hits10_nv = np.mean(_n_v_ranks <= 10.0)
    mrr_r = np.mean(1.0 / all_r_ranks)
    hits1_r = np.mean(all_r_ranks <= 1.0)
    hits3_r = np.mean(all_r_ranks <= 3.0)
    hits5_r = np.mean(all_r_ranks <= 5.0)
    hits10_r = np.mean(all_r_ranks <= 10.0)
    mrr_ht = np.mean(1.0 / all_ht_ranks)
    hits1_ht = np.mean(all_ht_ranks <= 1.0)
    hits3_ht = np.mean(all_ht_ranks <= 3.0)
    hits5_ht = np.mean(all_ht_ranks <= 5.0)
    hits10_ht = np.mean(all_ht_ranks <= 10.0)
    eval_result = {
        "entity": {
            "mrr": mrr_ent,
            "hits1": hits1_ent,
            "hits3": hits3_ent,
            "hits5": hits5_ent,
            "hits10": hits10_ent,
        },
        "relation": {
            "mrr": mrr_rel,
            "hits1": hits1_rel,
            "hits3": hits3_rel,
            "hits5": hits5_rel,
            "hits10": hits10_rel,
        },
        "ht": {
            "mrr": mrr_ht,
            "hits1": hits1_ht,
            "hits3": hits3_ht,
            "hits5": hits5_ht,
            "hits10": hits10_ht,
        },
        "2-ht": {
            "mrr": mrr_2ht,
            "hits1": hits1_2ht,
            "hits3": hits3_2ht,
            "hits5": hits5_2ht,
            "hits10": hits10_2ht,
        },
        "n-ht": {
            "mrr": mrr_nht,
            "hits1": hits1_nht,
            "hits3": hits3_nht,
            "hits5": hits5_nht,
            "hits10": hits10_nht,
        },
        "r": {
            "mrr": mrr_r,
            "hits1": hits1_r,
            "hits3": hits3_r,
            "hits5": hits5_r,
            "hits10": hits10_r,
        },
        "2-r": {
            "mrr": mrr_2r,
            "hits1": hits1_2r,
            "hits3": hits3_2r,
            "hits5": hits5_2r,
            "hits10": hits10_2r,
        },
        "n-r": {
            "mrr": mrr_nr,
            "hits1": hits1_nr,
            "hits3": hits3_nr,
            "hits5": hits5_nr,
            "hits10": hits10_nr,
        },
        "n-a": {
            "mrr": mrr_na,
            "hits1": hits1_na,
            "hits3": hits3_na,
            "hits5": hits5_na,
            "hits10": hits10_na,
        },
        "n-v": {
            "mrr": mrr_nv,
            "hits1": hits1_nv,
            "hits3": hits3_nv,
            "hits5": hits5_nv,
            "hits10": hits10_nv,
        },
    }
    return eval_result
