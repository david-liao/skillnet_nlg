

TASK_INFO = {
    # Text Summarization
    'lcsts': {'expert_ids': [0, 2]},

    # Advertisement Generation
    'adgen': {'expert_ids': [0, 1, 3]},

    # Closed-Book Question Answering
    'matinf': {'expert_ids': [0, 1, 5]},

    # Dialogue Generation
    'kdconv': {'expert_ids': [0, 1, 4, 5]},

    # GEC
    'nlpcc': {'expert_ids': [0, 2]},

    # Paraphrase
    'pkupb': {'expert_ids': [0, 2]},
    'bdpp': {'expert_ids': [0, 2]},

    # Story Generation
    'outgen': {'expert_ids': [0, 1]},

    # Topic-to-Essay Generation
    't2e': {'expert_ids': [0, 1, 3]},
}

from .multitask_trainer import (
    MultitaskDataloader,
    DataLoaderWithTaskname,
    MultitaskTrainer,
)
from .tokenization_t5bert_fast import T5BertTokenizerFast
from .modeling_pathway import PathwayForConditionalGeneration
from .configuration_pathway import PathwayConfig
