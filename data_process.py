"""
@文件    :data_process.py
@时间    :2022/12/02 21:07:23
@作者    :周恒
@版本    :1.0
@说明    :
"""

from collections import OrderedDict
import collections
import json
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Sized,
    Tuple,
    Union,
)
from dataclasses import asdict, dataclass, field
import typing
import numpy as np
import torch
import six


def convert_to_unicode(text):
    """
    Convert `text` to Unicode (if it's not already), assuming utf-8 input.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")


class Vocabulary(object):
    """
    Vocabulary class.
    """

    def __init__(self, vocab_file, num_relations, num_entities):
        self.vocab: typing.OrderedDict[str, int] = self.load_vocab(vocab_file)
        self.inv_vocab: typing.OrderedDict[int, str] = {
            v: k for k, v in self.vocab.items()
        }
        self.num_relations = num_relations
        self.num_entities = num_entities
        # 2 for special tokens of [PAD] and [MASK]
        assert len(self.vocab) == self.num_relations + self.num_entities + 2, (
            "The vocabulary contains all relations and entities, "
            "as well as 2 special tokens: [PAD] and [MASK]."
        )

    def load_vocab(self, vocab_file) -> typing.OrderedDict[str, int]:
        """
        Load a vocabulary file into a dictionary.
        """
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def convert_by_vocab(self, vocab, items):
        """
        Convert a sequence of [tokens|ids] using the vocab.
        """
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def convert_tokens_to_ids(self, tokens) -> List[int]:
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids) -> List[str]:
        return self.convert_by_vocab(self.inv_vocab, ids)


@dataclass
class NaryExample:
    arity: int
    relation: str
    head: str
    tail: str
    auxiliary_info: OrderedDict = field(default_factory=OrderedDict)


def read_examples(input_file) -> List[NaryExample]:
    """
    Read a n-ary json file into a list of NaryExample.
    """
    examples, total_instance = [], 0
    with open(input_file, "r") as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            assert (
                "N" in obj.keys()
                and "relation" in obj.keys()
                and "subject" in obj.keys()
                and "object" in obj.keys()
            ), "There are 4 mandatory fields: N, relation, subject, and object."
            arity = obj["N"]
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()
                # store attributes in alphabetical order
                for attribute in sorted(obj.keys()):
                    if (
                        attribute == "N"
                        or attribute == "relation"
                        or attribute == "subject"
                        or attribute == "object"
                    ):
                        continue
                    # store corresponding values in alphabetical order
                    auxiliary_info[attribute] = sorted(obj[attribute])
            """
            if len(examples) % 1000 == 0:
                logger.debug("*** Example ***")
                logger.debug("arity: %s" % str(arity))
                logger.debug("relation: %s" % relation)
                logger.debug("head: %s" % head)
                logger.debug("tail: %s" % tail)
                if auxiliary_info:
                    for attribute in auxiliary_info.keys():
                        logger.debug("attribute: %s" % attribute)
                        logger.debug("value(s): %s" % " ".join(
                            [value for value in auxiliary_info[attribute]]))
            """

            example = NaryExample(
                arity=arity,
                relation=relation,
                head=head,
                tail=tail,
                auxiliary_info=auxiliary_info,
            )
            examples.append(example)
            total_instance += 2 * (arity - 2) + 3

    return examples 


def generate_ground_truth(
    ground_truth_path: str, vocabulary: Vocabulary, max_arity: int, max_seq_length: int
):
    """
    Generate ground truth for filtered evaluation.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, (
        "Each input sequence contains relation, head, tail, "
        "and max_aux attribute-value pairs."
    )

    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    all_examples = read_examples(ground_truth_path)
    for (example_id, example) in enumerate(all_examples):
        # get padded input tokens and ids
        rht = [example.relation, example.head, example.tail]
        aux_attributes = []
        aux_values = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_attributes.append(attribute)
                    aux_values.append(value)

        while len(aux_attributes) < max_aux:
            aux_attributes.append("[PAD]")
            aux_values.append("[PAD]")
        assert len(aux_attributes) == max_aux
        assert len(aux_values) == max_aux

        input_tokens = rht + aux_attributes + aux_values
        input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
        assert len(input_tokens) == max_seq_length
        assert len(input_ids) == max_seq_length

        # get target answer for each pos and the corresponding key
        for pos in range(max_seq_length):
            if input_ids[pos] == 0:
                continue
            key = " ".join(
                [str(input_ids[x]) for x in range(max_seq_length) if x != pos]
            )
            gt_dict[pos][key].append(input_ids[pos])

    return gt_dict


@dataclass
class NaryFeature:
    feature_id: int
    example_id: int
    mask_position: int
    mask_label: int
    mask_type: int
    arity: int
    input_tokens: List[str] = field(default_factory=list)
    input_ids: List[int] = field(default_factory=list)
    input_mask: List[int] = field(default_factory=list)


def convert_examples_to_features(
    examples: List[NaryExample],
    vocabulary: Vocabulary,
    max_arity: int,
    max_seq_length: int,
) -> List[NaryFeature]:
    """
    Convert a set of NaryExample into a set of NaryFeature. Each single
    NaryExample is converted into (2*(n-2)+3) NaryFeature, where n is
    the arity of the given example.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, (
        "Each input sequence contains relation, head, tail, "
        "and max_aux attribute-value pairs."
    )

    features = []
    feature_id = 0
    for (example_id, example) in enumerate(examples):
        # get original input tokens and input mask
        rht = [example.relation, example.head, example.tail]
        rht_mask = [1, 1, 1]

        aux_attributes = []
        aux_attributes_mask = []
        aux_values = []
        aux_values_mask = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_attributes.append(attribute)
                    aux_values.append(value)
                    aux_attributes_mask.append(1)
                    aux_values_mask.append(1)

        while len(aux_attributes) < max_aux:
            aux_attributes.append("[PAD]")
            aux_values.append("[PAD]")
            aux_attributes_mask.append(0)
            aux_values_mask.append(0)
        assert len(aux_attributes) == max_aux
        assert len(aux_values) == max_aux
        assert len(aux_attributes_mask) == max_aux
        assert len(aux_values_mask) == max_aux

        orig_input_tokens = rht + aux_attributes + aux_values
        orig_input_mask = rht_mask + aux_attributes_mask + aux_values_mask
        assert len(orig_input_tokens) == max_seq_length
        assert len(orig_input_mask) == max_seq_length

        # generate a feature by masking each of the tokens
        for mask_position in range(max_seq_length):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue
            mask_label = vocabulary.vocab[orig_input_tokens[mask_position]]
            mask_type = (
                -1 if mask_position == 0 or 2 < mask_position < max_aux + 3 else 1
            )

            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"
            input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
            assert len(input_tokens) == max_seq_length
            assert len(input_ids) == max_seq_length

            feature = NaryFeature(
                feature_id=feature_id,
                example_id=example_id,
                input_tokens=input_tokens,
                input_ids=input_ids,
                input_mask=orig_input_mask,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                arity=example.arity,
            )
            features.append(feature)
            feature_id += 1

    return features


class DataPreprocess:
    def __init__(self, max_arity=2, max_seq_length=3) -> None:
        self.max_arity = max_arity
        self.max_seq_length = max_seq_length

    def __call__(self, examples: List[NaryExample], vocabulary: Vocabulary) -> Any:
        features = convert_examples_to_features(
            examples=examples,
            vocabulary=vocabulary,
            max_arity=self.max_arity,
            max_seq_length=self.max_seq_length,
        )
        return features


class GranCollator:
    def __init__(self, max_arity=2, max_seq_length=3) -> None:
        self.max_arity = max_arity
        self.max_seq_length = max_seq_length

    def __call__(self, batch_features: List[NaryFeature]) -> Any:
        batch_input_ids = torch.tensor(
            [inst.input_ids for inst in batch_features], dtype=torch.long
        ).reshape([-1, self.max_seq_length])
        batch_input_mask = torch.tensor(
            [inst.input_mask for inst in batch_features], dtype=torch.float32
        ).reshape([-1, self.max_seq_length, 1])
        batch_mask_position = torch.tensor(
            [inst.mask_position for inst in batch_features], dtype=torch.long
        )
        batch_mask_label = torch.tensor(
            [inst.mask_label for inst in batch_features], dtype=torch.long
        ).reshape([-1])
        batch_mask_type = torch.tensor(
            [inst.mask_type for inst in batch_features], dtype=torch.long
        ).reshape([-1])
        
        return (
            batch_input_ids,
            batch_input_mask,
            batch_mask_position,
            batch_mask_label,
            batch_mask_type, 
            batch_features
        )
