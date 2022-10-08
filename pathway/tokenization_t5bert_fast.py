# coding=utf-8

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece

from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import Converter

# from tokenization_t5bert import T5BertTokenizer
from transformers import BertTokenizer


class T5BertConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # eos = str(self.original_tokenizer.eos_token)
        # eos_token_id = self.original_tokenizer.eos_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer (:class:`~transformers.tokenization_utils_base.PreTrainedTokenizer`):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerFast`.

    Return:
        A instance of :class:`~tokenizers.Tokenizer` to be used as the backend tokenizer of a
        :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerFast`
    """

    tokenizer_class_name = transformer_tokenizer.__class__.__name__
    # assert tokenizer_class_name == 'T5BertTokenizer'
    assert tokenizer_class_name == 'BertTokenizer'

    return T5BertConverter(transformer_tokenizer).converted()


class T5BertTokenizerFast(BertTokenizerFast):

    slow_tokenizer_class = BertTokenizer

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        self._init_pretrained_tokenizer_fast(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        self.do_lower_case = do_lower_case

    def _init_pretrained_tokenizer_fast(self, *args, **kwargs):
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
        from_slow = kwargs.pop("from_slow", False)

        if from_slow and slow_tokenizer is None and self.slow_tokenizer_class is None:
            raise ValueError(
                "Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you "
                "have sentencepiece installed."
            )

        if tokenizer_object is not None:
            fast_tokenizer = tokenizer_object
        elif fast_tokenizer_file is not None and not from_slow:
            # We have a serialization from tokenizers which let us directly build the backend
            fast_tokenizer = Tokenizer.from_file(fast_tokenizer_file)
        elif slow_tokenizer is not None:
            # We need to convert a slow tokenizer to build the backend
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        elif self.slow_tokenizer_class is not None:
            # We need to create and convert a slow tokenizer to build the backend
            slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        else:
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: \n"
                "(1) a `tokenizers` library serialization file, \n"
                "(2) a slow tokenizer instance to convert or \n"
                "(3) an equivalent slow tokenizer class to instantiate and convert. \n"
                "You need to have sentencepiece installed to convert a slow tokenizer to a fast one."
            )

        self._tokenizer = fast_tokenizer

        if slow_tokenizer is not None:
            kwargs.update(slow_tokenizer.init_kwargs)

        self._decode_use_source_tokenizer = False

        # We call this after having initialized the backend tokenizer because we update it.
        PreTrainedTokenizerBase.__init__(self, **kwargs)
