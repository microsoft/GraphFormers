# coding=utf-8
"""Tokenization classes for TuringNLRv3."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open

from transformers.tokenization_bert import BertTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
}


class TuringNLRv3Tokenizer(BertTokenizer):
    r"""
    Constructs a TuringNLRv3Tokenizer.
    :class:`~transformers.TuringNLRv3Tokenizer` is identical to BertTokenizer and runs end-to-end tokenization: punctuation splitting + wordpiece
    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_wordpiece_only=False
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)
