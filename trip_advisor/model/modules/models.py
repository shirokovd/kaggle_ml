from enum import Enum

from transformers import BertTokenizer, BertForSequenceClassification


class TokenizerType(Enum):
    bert_tokenizer = 'bert_tokenizer'


TOKENIZERS_CLASSES = {
    TokenizerType.bert_tokenizer: BertTokenizer
}


class ModelType(Enum):
    bert_for_sequence_classification = 'bert_for_sequence_classification'


MODELS_CLASSES = {
    ModelType.bert_for_sequence_classification: BertForSequenceClassification
}
