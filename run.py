"""
@文件    :run.py
@时间    :2022/12/03 00:50:14
@作者    :周恒
@版本    :1.0
@说明    :
"""
from ast import arg
import dataclasses
from datetime import datetime
import os
import sys
import pickle
import random
import json
from time import time
from typing import Any, Dict
from typing_extensions import Literal

import numpy as np
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from trainer import Trainer
import torch
import argparse
from model import (
    GranConfig,
    GranModel,
    BatchMetricsFunc,
    batch_cal_loss_func,
    batch_forward_func,
    MetricsCalFunc,
    get_optimizer,
)
from data_process import (
    Vocabulary,
    NaryExample,
    NaryFeature,
    DataPreprocess,
    GranCollator,
    read_examples,
)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + " Default: %(default)s.",
            **kwargs
        )


parser = argparse.ArgumentParser()
model_g = ArgumentGroup(parser, "model", "model and checkpoint configuration.")
model_g.add_arg("num_hidden_layers", int, 12, "Number of hidden layers.")
model_g.add_arg("num_attention_heads", int, 4, "Number of attention heads.")
model_g.add_arg("hidden_size", int, 256, "Hidden size.")
model_g.add_arg("intermediate_size", int, 512, "Intermediate size.")
model_g.add_arg("hidden_act", str, "gelu", "Hidden act.")
model_g.add_arg("hidden_dropout_prob", float, 0.1, "Hidden dropout ratio.")
model_g.add_arg("attention_dropout_prob", float, 0.1, "Attention dropout ratio.")
model_g.add_arg("initializer_range", float, 0.02, "Initializer range.")
model_g.add_arg("vocab_size", int, None, "Size of vocabulary.")
model_g.add_arg("num_relations", int, None, "Number of relations.")
model_g.add_arg(
    "num_edges",
    int,
    5,
    "Number of edge types, typically fixed to 5: no edge (0), relation-subject (1),"
    "relation-object (2), relation-attribute (3), attribute-value (4).",
)
model_g.add_arg("max_seq_len", int, None, "Max sequence length.")
model_g.add_arg("max_arity", int, None, "Max arity.")
model_g.add_arg(
    "entity_soft_label", float, 1.0, "Label smoothing rate for masked entities."
)
model_g.add_arg(
    "relation_soft_label", float, 1.0, "Label smoothing rate for masked relations."
)
model_g.add_arg(
    "weight_sharing",
    bool,
    True,
    "If set, share masked lm weights with node embeddings.",
)


train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("batch_size", int,1024, "Batch size.")
train_g.add_arg("epoch", int, 100, "Number of training epochs.")
train_g.add_arg("learning_rate", float, 5e-4, "Learning rate with warmup.")
train_g.add_arg("gradient_accumulate",int, 1, "gradient accumulate steps")
train_g.add_arg("main_device", int, 0, "main gpu device for training")
train_g.add_arg("device_ids", str, "0", "all available gpu devices for training")
train_g.add_arg("num_workers", int, 12, "processes for preparing batch data")

data_g = ArgumentGroup(
    parser, "data", "Data paths, vocab paths and data processing options."
)
data_g.add_arg("train_file", str, None, "Data for training.")
data_g.add_arg("predict_file", str, None, "Data for prediction.")
data_g.add_arg("ground_truth_path", str, None, "Path to ground truth.")
data_g.add_arg("vocab_path", str, None, "Path to vocabulary.")


if __name__ == "__main__":
    set_random_seed(3709)
    args = parser.parse_args()
    dev = torch.device(args.main_device)
    device_ids = list(map(lambda x: int(x), args.device_ids.split(",")))
    output_dir="output/{}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    epochs = args.epoch
    gradient_accumulate = args.gradient_accumulate
    
    config=GranConfig(
        voc_size=args.vocab_size,
        n_relation=args.num_relations,
        n_edge=args.num_edges,
        max_seq_len=args.max_seq_len,
        max_arity=args.max_arity,
        n_layer=args.num_hidden_layers,
        n_head=args.num_attention_heads,
        emb_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        hidden_act=args.hidden_act,
        prepostprocess_dropout=args.hidden_dropout_prob,
        attention_dropout=args.attention_dropout_prob,
        initializer_range=args.initializer_range,
        e_soft_label=args.entity_soft_label,
        r_soft_label=args.relation_soft_label
    )
    with open("config.json","w") as f:
        json.dump(config,f,default=dataclasses.asdict,indent=4,ensure_ascii=False)
    
    train_examples=read_examples(args.train_file)
    valid_examples=read_examples(args.predict_file)
    vocabulary=Vocabulary(args.vocab_path,num_relations=config.n_relation,num_entities=config.voc_size-config.n_relation-2)

    data_preprocessor=DataPreprocess(config.max_arity,config.max_seq_len)
    train_dataset=data_preprocessor(train_examples,vocabulary)
    valid_dataset=data_preprocessor(valid_examples,vocabulary)

    batch_metrics_func=BatchMetricsFunc(args.ground_truth_path,vocabulary,config.max_arity,config.max_seq_len)
    metrics_cal_func=MetricsCalFunc()

    collator=GranCollator(config.max_arity,config.max_seq_len)
    train_dataset_sampler=RandomSampler(train_dataset)
    valid_dataset_sampler=SequentialSampler(valid_dataset)

    model=GranModel(config,args.weight_sharing)
    optimizer=get_optimizer(model,learning_rate)
    
    if len(device_ids)>1:
        model=torch.nn.parallel.DataParallel(model,device_ids)

    trainer=Trainer(
        model=model,
            optimizer=optimizer,
            output_dir=output_dir,
            training_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=None,
            metrics_key="entity",
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            batch_forward_func=batch_forward_func,
            batch_cal_loss_func=batch_cal_loss_func,
            batch_metrics_func=batch_metrics_func,
            metrics_cal_func=metrics_cal_func,
            collate_fn=collator,
            device=dev,
            train_dataset_sampler=train_dataset_sampler,
            valid_dataset_sampler=valid_dataset_sampler,
            valid_step=1,
            start_epoch=0,
            gradient_accumulate=gradient_accumulate,
            save_model=True,
            save_model_steps=5
    )
    trainer.train()
