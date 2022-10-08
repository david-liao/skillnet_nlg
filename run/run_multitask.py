#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import torch
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset

import transformers

from transformers import (
    BertTokenizerFast,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    MBartTokenizerFast,
    BartForConditionalGeneration,
    MBartForConditionalGeneration,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed
)

from torch.utils.data.dataset import ConcatDataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint

sys.path.append(os.getcwd())
from pathway import (
    MultitaskTrainer,
    T5BertTokenizerFast,
    PathwayForConditionalGeneration,
    PathwayConfig,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    'bart': [BertTokenizerFast, BartForConditionalGeneration, 'cache'],
    't5': [T5BertTokenizerFast, T5ForConditionalGeneration, 't5_cache'],
    'moe': [BertTokenizerFast, PathwayForConditionalGeneration, 'cache'],
    'pathway': [BertTokenizerFast, PathwayForConditionalGeneration, 'cache'],
}
DATA_CONFIG = {
    'lcsts': 'summarization.py',
    'adgen': 'summarization.py',
    'kdconv': 'summarization.py',
    'csl': 'summarization.py',
    'matinf': 'summarization.py',
    'mask': 'summarization.py',
    'nlpcc': 'summarization.py',
    'nlpcc-raw': 'summarization.py',
    'cwn': 'summarization.py',
    'math': 'summarization.py'
}

# global flag for logging once
log_feature = True
log_example = True
log_prediction = True


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(metadata={"help": "Model type"})
    encoder_moe_layers: Optional[int] = field(
        default=12,
        metadata={"help": "The number of encoder moe layers in the pathway model."},
    )
    decoder_moe_layers: Optional[int] = field(
        default=12,
        metadata={"help": "The number of decoder moe layers in the pathway model."},
    )
    topk_experts: Optional[int] = field(
        default=2,
        metadata={"help": "The number of experts in the baseline moe model. -1 is for pathway model."},
    )
    expert_ids: Optional[List[int]] = field(
        default=None, metadata={"help": "The list of experts for skills."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the dataset to use (via the datasets library)."},
    )
    # data_name: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "The name of the dataset to use (via the datasets library)."},
    # )
    datasets: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of datasets to load for multitask training."}
    )
    train_file: Optional[str] = field(
        default='train.jsonl',
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default='dev.jsonl',
        metadata={"help": "An optional input evaluation data file to evaluate the metrics (rouge) on"
                  "(a jsonlines or csv file)."},
    )
    test_file: Optional[str] = field(
        default='test.jsonl',
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on"
            "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help":
            "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help":
            "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                  "value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                  "value if set."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                  "value if set."},
    )
    num_beams: Optional[int] = field(
        default=3,
        metadata={
            "help":
            "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."},
    )
    temperature: float = field(default=1.0, metadata={"help": "Temperature sampling for multitask training."})


# Data collator
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    model_name: str = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        if self.model_name == 't5-pegasus' or 'bert' in self.model_name or self.model_name == 'bart-large':
            # This condition will be removed in the future.
            raise NotImplementedError
        else:
            labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
            # same length to return tensors.
            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )

            features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # prepare decoder_input_ids
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        global log_feature
        if log_feature:
            log_strs = ["*** Feature ***"]
            for k, v in features.items():
                log_strs.append(k + ':\n  ' + str(v[0]))
            logger.info('\n'.join(log_strs))

            log_feature = False

        return features


# @light_init(params={"training_framework": "pytorch_ddp"})
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}," +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_explicit_format()

    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # comment below condition because we save train_x.log in output_dir which make os.listdir(output_dir) always > 0
        # if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        #     raise ValueError(
        #         f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        #         "Use --overwrite_output_dir to overcome."
        #     )
        # elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model_name = model_args.model_type
    print("model_name", model_name)
    tokenizer_class, model_class, cache_dir_name = MODEL_CONFIG[model_name]

    # Get the datasets
    raw_datasets_dict = {}
    for data_name in data_args.datasets:
        # assert data_name in DATA_CONFIG
        dataset_file = DATA_CONFIG.get(data_name, 'summarization.py')

        data_files = {}
        if training_args.do_train:
            train_file = os.path.join(data_args.data_path, data_name, data_args.train_file)
            data_files["train"] = train_file
        if training_args.do_eval:
            validation_file = os.path.join(data_args.data_path, data_name, data_args.validation_file)
            data_files["validation"] = validation_file
        if training_args.do_predict:
            test_file = os.path.join(data_args.data_path, data_name, data_args.test_file)
            data_files["test"] = test_file

        data_path = os.path.join(data_args.data_path, 'datasets', dataset_file)
        cache_dir = os.path.join(data_args.data_path, data_name, cache_dir_name)
        raw_datasets_dict[data_name] = load_dataset(data_path, data_files=data_files, cache_dir=cache_dir)

    # Load pretrained model and tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)
    config = None
    if model_name in ("moe", "pathway"):
        if model_name == "moe" and model_args.topk_experts <= 0:
            raise ValueError("--topk_experts must great than 0 for moe model")
        if model_name == "pathway" and model_args.topk_experts != -1:
            raise ValueError("--topk_experts must equal -1 for pathway model")
        config = PathwayConfig.from_pretrained(model_args.model_name_or_path)
        config.topk_experts = model_args.topk_experts
        config.encoder_moe_layers = model_args.encoder_moe_layers
        config.encoder_moe_layers = model_args.encoder_moe_layers
    model = model_class.from_pretrained(model_args.model_name_or_path, config=config)

    def model_size(model):
        para_num = sum([np.prod(list(p.size())) for p in model.parameters()])
        return para_num

    def get_gen_kwargs(model_name, tokenizer, model):
        if model.config.pad_token_id != tokenizer.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(
                f"fix model.config.pad_token_id: {model.config.pad_token_id} to tokenizer.pad_token_id: {tokenizer.pad_token_id}")

        if model_name in ('bart', 'moe', 'pathway'):
            decoder_start_token_id = tokenizer.sep_token_id
            eos_token_id = tokenizer.sep_token_id
        elif model_name == 't5':
            decoder_start_token_id = tokenizer.pad_token_id
            eos_token_id = tokenizer.eos_token_id
        else:
            logger.info(f"Not support model: {model_name}")
            exit()

        gen_kwargs = {
            "max_length": data_args.val_max_target_length,
            "num_beams": data_args.num_beams,
            "early_stopping": True,
            "decoder_start_token_id": decoder_start_token_id,
            "eos_token_id": eos_token_id
        }
        logger.info(f"Generation paramerters: {gen_kwargs}")

        logger.info(f"Train with model name: {model_name}\n\
            tokenizer: {tokenizer.__class__}\n\
            model: {model.__class__}, model param num: {model_size(model) / 1e6:.2f} M\n\
            decoder_start_token: {tokenizer.convert_ids_to_tokens(decoder_start_token_id)}, eos_token: {tokenizer.convert_ids_to_tokens(eos_token_id)}")
        return gen_kwargs

    gen_kwargs = get_gen_kwargs(model_name, tokenizer, model)

    model.resize_token_embeddings(len(tokenizer))
    logger.info(model)
    # prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory")

    def preprocess_function(examples):
        inputs = examples['text']
        inputs = [task_prefix + inp for inp in inputs]
        targets = examples['summary']

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        model_inputs.pop('token_type_ids')

        global log_example
        if log_example:
            i = 0
            example = {
                'text': inputs[i],
                'input_ids': model_inputs["input_ids"][i],
                'input_ids_decode': tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][i]),
                'summary': targets[i],
                'labels': model_inputs["labels"][i],
                'labels_decode': tokenizer.convert_ids_to_tokens(model_inputs["labels"][i])
            }

            log_strs = ["*** Input Example ***"]
            for k, v in example.items():
                log_strs.append(k + ':\n  ' + str(v))
            logger.info('\n'.join(log_strs))

            log_example = False

        return model_inputs

    def before_data_mapping(cur_task):
        if training_args.local_rank != -1 and training_args.process_index > 0:
            logger.warning(f"Waiting for main process to perform the mapping. Task: {cur_task}, Process index: "
                           f"{training_args.process_index}, Local process index: {training_args.local_process_index}")
            torch.distributed.barrier()

    def after_data_mapping(cur_task):
        if training_args.local_rank != -1 and training_args.process_index == 0:
            logger.warning(f"Loading results from main process. Task: {cur_task}, Process index: "
                           f"{training_args.process_index}, Local process index: {training_args.local_process_index}")
            torch.distributed.barrier()

    if training_args.do_train:
        train_dataset_dict = {}
        for task, raw_datasets in raw_datasets_dict.items():
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            column_names = raw_datasets["train"].column_names
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))

            task_prefix = task + ': '
            before_data_mapping(task)
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on {task} train dataset",
            )
            after_data_mapping(task)
            train_dataset_dict[task] = train_dataset

    if training_args.do_eval:
        eval_dataset_dict = {}
        for task, raw_datasets in raw_datasets_dict.items():
            max_target_length = data_args.val_max_target_length  # used by preprocess_function()
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            column_names = raw_datasets["validation"].column_names
            eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

            task_prefix = task + ': '
            before_data_mapping(task)
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=F"Running tokenizer on {task} validation dataset",
            )
            after_data_mapping(task)
            eval_dataset_dict[task] = eval_dataset
        # eval_dataset = ConcatDataset(eval_dataset_dict.values())

    if training_args.do_predict:
        predict_dataset_dict = {}
        for task, raw_datasets in raw_datasets_dict.items():
            max_target_length = data_args.val_max_target_length  # # used by preprocess_function()
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            column_names = raw_datasets["test"].column_names
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

            task_prefix = task + ': '
            before_data_mapping(task)
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            after_data_mapping(task)
            predict_dataset_dict[task] = predict_dataset
        # predict_dataset = ConcatDataset(predict_dataset_dict.values())

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        model_name=model_name,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    def postprocess_text(preds, refs):
        preds = [pred.strip() for pred in preds]
        refs = [ref.strip() for ref in refs]

        # preds = [re.sub(' +', ' ', ' '.join(list(pred.strip()))) for pred in preds]
        # refs = [re.sub(' +', ' ', ' '.join(list(ref.strip()))) for ref in refs]

        return preds, refs

    from rouge import Rouge

    rouge = Rouge()

    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    import jieba

    def bleu(data):
        """
        compute rouge score
        Args:
            data (list of dict including reference and candidate):
        Returns:
                res (dict of list of scores): rouge score
        """

        res = {}
        for i in range(1, 5):
            res["bleu-%d" % i] = []

        for tmp_data in data:
            origin_candidate = tmp_data['candidate']
            origin_reference = tmp_data['reference']
            assert isinstance(origin_candidate, str)
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]

            for i in range(1, 5):
                res["bleu-%d" % i].append(sentence_bleu(references=[r.strip().split() for r in origin_reference],
                                                        hypothesis=origin_candidate.strip().split(),
                                                        weights=tuple([1. / i for j in range(i)]),
                                                        smoothing_function=SmoothingFunction().method3))

        for key in res:
            res[key] = np.mean(res[key])

        return res

    def proline(line):
        return " ".join([w for w in jieba.cut("".join(line.strip().split()))])

    def compute_score(preds, refs):
        score = {}
        if training_args.metric_for_best_model == 'rouge-l':
            try:
                rouge_score = rouge.get_scores(preds, refs, avg=True, ignore_empty=True)
                rouge_score = {key: value['f'] * 100 for key, value in rouge_score.items()}
            except Exception as e:
                logger.info(f'the rouge score is invalid, {str(e)}')
                rouge_score = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

            score.update(rouge_score)

        elif training_args.metric_for_best_model == 'bleu':
            preds = [pred.split() for pred in preds]
            refs = [[ref.split()] for ref in refs]
            try:
                bleu_score = corpus_bleu(refs, preds, smoothing_function=SmoothingFunction().method3)
            except ZeroDivisionError as _:
                logger.info('the bleu score is invalid')
                bleu_score = 0

            score['bleu'] = bleu_score * 100

        elif training_args.metric_for_best_model == 'sentence_bleu':
            bleu_score = 0
            for pred, ref in zip(preds, refs):
                bleu_score += sentence_bleu([ref.split()], pred.split(), smoothing_function=SmoothingFunction().method2)

            score['sentence_bleu'] = bleu_score / len(preds) * 100

        elif training_args.metric_for_best_model == 'acc':
            def compute_value(s):
                s = s.replace('. ', '.').replace('%', '* 0.01').replace('[', '(').replace(']', ')')
                return eval(s)

            def compute_acc(pred, ref):
                if (pred == ref):
                    return 1

                try:
                    res = compute_value(pred) - compute_value(ref)
                except Exception as e:
                    # print(pred, e)
                    return 0

                if abs(res) < 1e-4:
                    return 1
                else:
                    return 0

            acc = sum([compute_acc(pred, ref) for pred, ref in zip(preds, refs)]) / len(preds)
            score['acc'] = acc

        elif training_args.metric_for_best_model == 'bleu-2':
            assert len(data_args.datasets) == 1
            task = data_args.datasets[0]
            assert task in ('t2e', 'outgen')

            if task in ('t2e'):
                preds_lst = [pred.split() for pred in preds]
                refs_lst = [[ref.split()] for ref in refs]
                for i in range(1, 5):
                    weights = (1. / i,) * i
                    bleu_score = corpus_bleu(refs_lst, preds_lst, weights=weights,
                                             smoothing_function=SmoothingFunction().method3)
                    score[f'bleu-{i}'] = bleu_score * 100
            elif task in ('outgen'):
                eval_data = [{"reference": proline(ref), "candidate": proline(pred)} for ref, pred in zip(refs, preds)]
                bleu_score = bleu(eval_data)
                bleu_score = {key: value * 100 for key, value in bleu_score.items()}
                score.update(bleu_score)

        return score

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        post_preds, post_labels = postprocess_text(decoded_preds, decoded_labels)

        # log prediction
        global log_prediction
        if log_prediction:
            i = 0
            example = {
                'predict_ids': preds[i].tolist(),
                'predict_ids_convert': tokenizer.convert_ids_to_tokens(preds[i]),
                'predict_ids_decode': post_preds[i],
                'label_ids': labels[i].tolist(),
                'labels_ids_convert': tokenizer.convert_ids_to_tokens(labels[i]),
                'labels_ids_decode': post_labels[i]
            }

            log_strs = ["*** Prediction Example ***"]
            for k, v in example.items():
                log_strs.append(k + ':\n  ' + str(v))
            logger.info('\n'.join(log_strs))

            log_prediction = False

        result = compute_score(post_preds, post_labels)

        prediction_lens = [len(pred.split()) for pred in decoded_preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    # Initialize our Trainer
    trainer = MultitaskTrainer(model=model,
                               args=training_args,
                               train_dataset=train_dataset_dict if training_args.do_train else None,
                               eval_dataset=eval_dataset_dict if training_args.do_eval else None,
                               tokenizer=tokenizer,
                               data_collator=data_collator,
                               compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                               gen_kwargs=gen_kwargs,
                               temperature=data_args.temperature,
                               expert_ids=model_args.expert_ids)

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        train_dataset_len = sum(len(dataset) for dataset in train_dataset_dict.values())
        max_train_samples = data_args.max_train_samples * len(
            train_dataset_dict) if data_args.max_train_samples else train_dataset_len
        metrics["train_samples"] = min(max_train_samples, train_dataset_len)

        metrics = {k: round(v, 2) for k, v in metrics.items()}
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(metric_key_prefix="evaluate")
        eval_dataset_len = sum(len(dataset) for dataset in eval_dataset_dict.values())
        max_eval_samples = data_args.max_eval_samples * len(
            eval_dataset_dict) if data_args.max_eval_samples else eval_dataset_len
        metrics["eval_samples"] = min(max_eval_samples, eval_dataset_len)

        metrics = {k: round(v, 2) for k, v in metrics.items()}
        trainer.log_metrics("evaluate", metrics)
        trainer.save_metrics("evaluate", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        metrics = trainer.predict(predict_dataset_dict, metric_key_prefix="predict")
        # metrics = predict_results.metrics
        predict_dataset_len = sum(len(dataset) for dataset in predict_dataset_dict.values())
        max_predict_samples = data_args.max_predict_samples * len(
            predict_dataset_dict) if data_args.max_predict_samples else predict_dataset_len
        metrics["predict_samples"] = min(max_predict_samples, predict_dataset_len)

        metrics = {k: round(v, 2) for k, v in metrics.items()}
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()

