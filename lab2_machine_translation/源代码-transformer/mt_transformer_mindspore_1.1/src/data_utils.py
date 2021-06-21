# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Create training instances for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
from easydict import EasyDict as edict
import logging
import numpy as np

from mindspore.mindrecord import FileWriter
import src.tokenization as tokenization

class SampleInstance():
    """A single sample instance (sentence pair)."""

    def __init__(self, source_sos_tokens, source_eos_tokens, target_sos_tokens, target_eos_tokens):
        self.source_sos_tokens = source_sos_tokens
        self.source_eos_tokens = source_eos_tokens
        self.target_sos_tokens = target_sos_tokens
        self.target_eos_tokens = target_eos_tokens

    def __str__(self):
        s = ""
        s += "source sos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.source_sos_tokens]))
        s += "source eos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.source_eos_tokens]))
        s += "target sos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.target_sos_tokens]))
        s += "target eos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.target_eos_tokens]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_file(writer, instance, tokenizer, max_seq_length):
    """Create files from `SampleInstance`s."""

    def _convert_ids_and_mask(input_tokens):
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, input_mask

    source_sos_ids, source_sos_mask = _convert_ids_and_mask(instance.source_sos_tokens)
    source_eos_ids, source_eos_mask = _convert_ids_and_mask(instance.source_eos_tokens)
    target_sos_ids, target_sos_mask = _convert_ids_and_mask(instance.target_sos_tokens)
    target_eos_ids, target_eos_mask = _convert_ids_and_mask(instance.target_eos_tokens)
    
    features = collections.OrderedDict()
    features["source_sos_ids"] = np.asarray(source_sos_ids,dtype=np.int32)
    features["source_sos_mask"] = np.asarray(source_sos_mask,dtype=np.int32)
    features["source_eos_ids"] = np.asarray(source_eos_ids,dtype=np.int32)
    features["source_eos_mask"] = np.asarray(source_eos_mask,dtype=np.int32)
    features["target_sos_ids"] = np.asarray(target_sos_ids,dtype=np.int32)
    features["target_sos_mask"] = np.asarray(target_sos_mask,dtype=np.int32)
    features["target_eos_ids"] = np.asarray(target_eos_ids,dtype=np.int32)
    features["target_eos_mask"] = np.asarray(target_eos_mask,dtype=np.int32)
    writer.write_raw_data([features])
    return features

def create_training_instance(source_words, target_words, max_seq_length):
    """Creates `SampleInstance`s for a single sentence pair."""
    EOS = "</s>"
    SOS = "<s>"

    source_sos_tokens = [SOS] + source_words
    source_eos_tokens = source_words + [EOS]
    target_sos_tokens = [SOS] + target_words
    target_eos_tokens = target_words + [EOS]

    instance = SampleInstance(
        source_sos_tokens=source_sos_tokens,
        source_eos_tokens=source_eos_tokens,
        target_sos_tokens=target_sos_tokens,
        target_eos_tokens=target_eos_tokens)
    return instance