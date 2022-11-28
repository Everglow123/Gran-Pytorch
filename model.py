"""
@文件    :model.py
@时间    :2022/11/17 00:06:29
@作者    :周恒
@版本    :1.0
@说明    :
"""





from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch import Tensor
from torch.nn import Dropout, Embedding, LayerNorm, Module, Sequential,Linear,MultiheadAttention
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.models.bert import BertConfig, BertForMaskedLM, BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta import (RobertaConfig, RobertaForMaskedLM,
                                         RobertaModel)
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer


@dataclass
class GranConfig:
    voc_size:int
    n_relation:int
    n_edge:int
    max_seq_len:int
    max_arity:int
    n_layer:int=12
    n_head:int=4
    emb_size:int=256
    intermediate_size:int=512
    hidden_act:str="gelu"
    prepostprocess_dropout:float=0.1
    attention_dropout:float=0.1
    initializer_range:float=0.02
    e_soft_label:float=1.0
    r_soft_label:float=1.0

def gen_act_func(act:str)->Callable[[Tensor],Tensor]:
    act=act.lower()
    res=eval("torch.nn.functional."+act)
    return res

class GranPrePostProcessLayer(Module):
    def __init__(self,shape:Sequence[int],process_cmd:str,dropout_rate:Optional[float]=None) -> None:
        super().__init__()
        layers:List[Module]=[]
        for cmd in process_cmd:
            if cmd=='n':
                layer=LayerNorm(shape,dtype=torch.float32)
                layers.append(layer)
            elif cmd=='d':
                rate=0.5
                if dropout_rate:
                    rate=dropout_rate
                    layer=Dropout(rate)
                    layers.append(layer)
        self.layers:Sequential=Sequential(*layers)
    def forward(self,input:Tensor,prev=None):
        if prev is not None:
            input=prev+input
        return self.layers(input)
class GranPositionWiseFeedForward(Module):
    def __init__(self,emb_size:int,d_inner_hid:int,dropout_rate:float,hidden_act=Callable[[Tensor],Tensor]) -> None:
        super().__init__()
        self.emb_size=emb_size
        self.d_inner_hid=d_inner_hid
        self.fc1=Linear(emb_size,d_inner_hid)
        self.act=hidden_act
        self.dropout=Dropout(dropout_rate)
        self.fc2=Linear(d_inner_hid,emb_size)
    def forward(self,x:Tensor):
        hidden=self.fc1(x)
        hidden=self.act(hidden)
        hidden=self.dropout(hidden)
        out=self.fc2(hidden)
        return out
class GranMultiHeadAttention(Module):
    def __init__(self,config:GranConfig,d_key:int,d_value:int,d_model:int,n_head:int=1,dropout_rate=0.0) -> None:
        super().__init__()
        self.config=config
        self.d_key=d_key
        self.d_value=d_value
        self.d_model=d_model
        self.n_head=n_head
        self.dropout_rate=dropout_rate
        self.query_fc=Linear(self.config.emb_size,self.d_key*self.n_head)
        self.key_fc=Linear(self.config.emb_size,self.d_key*self.n_head)
        self.value_fc=Linear(self.config.emb_size,self.d_key*self.n_head)
        self.proj_fc=Linear(self.config.emb_size,self.d_model)
    def compute_qkv(self,queries:Tensor,
        keys:Tensor,
        values:Tensor)->Tuple[Tensor,...]:
        # queries: batch_size*max_seq_len*emb_size
        # q: batch_size*max_seq_len*(d_key*self.n_head)
        q=self.query_fc(queries)
        k=self.key_fc(keys)
        v=self.value_fc(values)
        return q,k,v
    def split_heads(self,x:Tensor):
        hidden_size=x.shape[-1] 
        reshaped=torch.reshape(x,shape=[x.shape[0],x.shape[1],self.n_head,hidden_size//self.n_head])
        res=torch.transpose(reshaped,1,2)
        return res
    def edge_aware_self_attention(
        self,
        q:Tensor,#[B, N, M, H]
        k:Tensor,#[B, N, M, H]
        v:Tensor,#[B, N, M, H]
        edges_k:Tensor,#[M, M, H]
        edges_v:Tensor,#[M, M, H]
        attn_bias:Tensor,#[B, N, M, M]
    ):
        if not (len(q.shape) == len(k.shape) == len(v.shape) == 4):
            raise ValueError("Input q, k, v should be 4-D Tensors.")
        if not (len(edges_k.shape) == len(edges_v.shape) == 3):
            raise ValueError(
                "Input edges_k and edges_v should be 3-D Tensors.")
        scaled_q=q*(self.d_key**-0.5)
        product=torch.matmul(scaled_q,k.transpose(2,3))
        if edges_k and edges_v:
            scaled_q=scaled_q.permute([2,0,1,3])
            scaled_q=torch.reshape(scaled_q,shape=[0,-1,scaled_q.shape[3]])
            edge_bias=torch.matmul(scaled_q,edges_k.transpose(2,3))
            edge_bias=torch.reshape(edge_bias,[edge_bias.shape[0],-1,edge_bias.shape[1],edge_bias.shape[2]])
            edge_bias=torch.permute(edge_bias,[1,2,0,3])
            product+=edge_bias
        if attn_bias:
            product+=attn_bias
        weights=F.softmax(product)
        if self.dropout_rate:
            weights=F.dropout(weights,self.dropout_rate)
        out=torch.matmul(weights,v)
        if edges_k and edges_v:
            reshaped_weights=weights.permute([2,0,1,3])
            reshaped_weights=reshaped_weights.reshape(
                [reshaped_weights.shape[0],-1,reshaped_weights.shape[3]]
            )
            edge_bias=torch.matmul(reshaped_weights,edges_v)
            edge_bias=edge_bias.reshape(shape=[edge_bias.shape[0],-1,q.shape[1],q.shape[3]])
            edge_bias=edge_bias.permute([1,2,0,3])
            out+=edge_bias
        return out
    def combine_heads(self,x:Tensor):
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        trans_x=x.permute([0,2,1,3])
        trans_x=trans_x.reshape([trans_x.shape[0],trans_x.shape[1],-1])
        return trans_x
    def forward(
        self,
        queries:Optional[Tensor],
        keys:Optional[Tensor],
        values:Optional[Tensor],
        edges_key:Tensor,
        edges_value:Tensor,
        attn_bias:Tensor, 
        cache:Optional[Dict[str,Tensor]]=None
    ):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")
        q,k,v=self.compute_qkv(queries,keys,values)

        if cache is not None:
            old_shape=cache['k'].shape
            k=cache['k']=torch.cat([
                torch.reshape(cache['k'],[old_shape[0],old_shape[1],self.d_model]),k
            ],dim=1)
            old_shape=cache['v'].shape
            v=cache['v']=torch.cat([
                torch.reshape(cache['v'],[old_shape[0],old_shape[1],self.d_model]),v
            ],dim=1)
        q=self.split_heads(q)
        k=self.split_heads(k)
        v=self.split_heads(v)
        ctx_multiheads=self.edge_aware_self_attention(
            q,k,v,edges_key,edges_value,attn_bias 
        )
        out=self.combine_heads(ctx_multiheads)
        proj_out=self.proj_fc(out)
        return proj_out
        
class GranEncoderLayer(Module):
    def __init__(
        self,
        config:GranConfig,
        d_key:int,
        d_value:int,
        d_model:int,
        n_head:int,
        prepostprocess_dropout:float,
        attention_dropout:float,
        relu_dropout:float,
        hidden_act:Callable[[Tensor],Tensor],
        preprocess_cmd="",
        postprocess_cmd="dan",
    ) -> None:
        super().__init__()
        self.post_process_layer1=GranPrePostProcessLayer([config.max_seq_len,config.emb_size],preprocess_cmd,prepostprocess_dropout)
        self.multi_head_attention=GranMultiHeadAttention(
            config,d_key,d_value,d_model,n_head,attention_dropout
        )
        self.post_process_layer2=GranPrePostProcessLayer([config.max_seq_len,config.emb_size],postprocess_cmd,prepostprocess_dropout)
        self.positionwise_feed_forward=GranPositionWiseFeedForward(
            config.emb_size,config.intermediate_size,relu_dropout,hidden_act
        )
        self.post_process_layer3=GranPrePostProcessLayer([config.max_seq_len,config.emb_size],postprocess_cmd,prepostprocess_dropout)

    def forward(
        self,
        enc_input:Tensor,
        edges_key:Tensor,
        edges_value:Tensor,
        atten_bias:Tensor
    ):
        enc_input=self.post_process_layer1(enc_input)
        attn_output:Tensor=self.multi_head_attention(enc_input,None,None,edges_key,edges_value,atten_bias)
        attn_output=self.post_process_layer2(attn_output,enc_input)
        ffd_output=self.positionwise_feed_forward(attn_output)
        res=self.post_process_layer3(ffd_output,attn_output)
        return res

class GranEncoder(Module):
    def __init__(
        self,
        config:GranConfig,
        n_layer:int,
        d_key:int,
        d_value:int,
        d_model:int,
        n_head:int,
        d_inner_hid:int,
        prepostprocess_dropout:float,
        attention_dropout:float,
        relu_dropout:float,
        hidden_act=F.relu,
        preprocess_cmd="",
        postprocess_cmd="dan",
    ) -> None:
        super().__init__()
        self.n_layer=n_layer
        layers=[]
        for i in range(n_layer):
            layers.append(GranEncoderLayer(
                config,d_key,d_value,d_model,n_head,prepostprocess_dropout,attention_dropout,relu_dropout,
                hidden_act,
                preprocess_cmd,postprocess_cmd
            ))
        self.layers=torch.nn.ModuleList(layers)
        self.pre_post_process_layer=GranPrePostProcessLayer(
            [config.max_seq_len,config.emb_size],preprocess_cmd,prepostprocess_dropout
        )
        
    def forward(self,
        enc_input:Tensor,
        edges_key:Tensor,
        edges_value:Tensor,
        attn_bias:Tensor,
    ):
        for i in range(self.n_layer):
            layer:GranEncoderLayer=self.layers[i]
            enc_output=layer(enc_input,edges_key,edges_value,attn_bias)
            enc_input=enc_output
        enc_output=self.pre_post_process_layer(enc_input)
        return enc_output

class GranModel(Module):
    def __init__(
        self,
        mlm_name_or_path:str,
        config:GranConfig
    ) -> None:
        super().__init__()
        self.mlm:Union[BertModel,RobertaModel]=AutoModel.from_pretrained(
            mlm_name_or_path)
        self.config=config
        self.pre_post_process_layer1=GranPrePostProcessLayer(
            [self.config.max_seq_len,self.config.emb_size],
            "nd",self.config.prepostprocess_dropout
        )
        self.ent_embedding=Embedding(self.config.voc_size,self.config.emb_size)
        self.edge_key_embedding=Embedding(self.config.n_edge,self.config.emb_size//self.config.n_head)
        self.edge_value_embedding=Embedding(self.config.n_edge,self.config.emb_size//self.config.n_head)
        self.encoder=GranEncoder(
            config=config,
            n_layer=config.n_layer,
            d_key=config.emb_size//config.n_head,
            d_value=config.emb_size//config.n_head,
            d_model=config.emb_size,
            d_inner_hid=config.intermediate_size,
            prepostprocess_dropout=config.prepostprocess_dropout,
            attention_dropout=config.attention_dropout,
            relu_dropout=0.0,
            hidden_act=gen_act_func(config.hidden_act),
            preprocess_cmd="",
            postprocess_cmd="dan"
        )

        self.mask_trans_feat_fc=Linear(self.config.emb_size,self.config.emb_size)
        self.mask_trans_feat_pre_process_layer=GranPrePostProcessLayer(shape=[config.max_seq_len,config.emb_size])
    def forward(
        self,
        input_ids:Tensor,# batch_size * max_seq_len
        input_mask:Tensor,
        edge_labels:Tensor, # max_seq_len * max_seq_len * 1,
        mask_pos:Tensor #[batch_size * 1]
    ):
        entity_emb=self.ent_embedding(input_ids)
        """batch_size*max_seq_size*hidden_size"""
        emb_out=self.pre_post_process_layer1(entity_emb)
        """(max_seq_len * max_seq_len) * 1"""
        edge_labels=edge_labels.reshape([-1,1])

        """(max_seq_len * max_seq_len) * (config.emb_size//self.config.n_head)"""
        edges_key:Tensor=self.edge_key_embedding(edge_labels)

        """(max_seq_len * max_seq_len) * (config.emb_size//self.config.n_head)"""
        edges_value:Tensor=self.edge_value_embedding(edge_labels)
        
        """(max_seq_len * max_seq_len) * 1"""
        edge_mask=torch.sign(edge_labels.float())
        edges_key=edges_key*edge_mask
        edges_value=edges_value*edge_mask
        edges_value=edges_value.reshape([self.config.max_seq_len,self.config.max_seq_len,-1])
        attn_mask=torch.matmul(input_mask,input_mask.transpose(1,2))
        attn_mask=1000000.0*(attn_mask-0.1)
        n_head_self_attn_mask=[attn_mask]*self.config.n_head
        n_head_self_attn_mask=torch.concat(n_head_self_attn_mask,dim=1)
        n_head_self_attn_mask=n_head_self_attn_mask.detach_()

        enc_out=self.encoder(emb_out,edges_key,edges_value,n_head_self_attn_mask)
        # [(batch_size*max_seq_len)*emb_size]
        reshaped_emb_out=torch.reshape(enc_out,[-1,self.config.emb_size])

        mask_feat=reshaped_emb_out[mask_pos]
        mask_trans_feat = self.mask_trans_feat_fc(mask_feat)
        mask_trans_feat=gen_act_func(self.config.hidden_act)(mask_trans_feat)



