from enum import Enum

from transformers import AdamW, get_linear_schedule_with_warmup


class OptimizerType(Enum):
    adam_w = 'adam_w'


OPTIMIZERS_CLASSES = {
    OptimizerType.adam_w: AdamW
}


class SchedulerType(Enum):
    linear_schedule_with_warmup = 'linear_schedule_with_warmup'


SCHEDULERS_CLASSES = {
    SchedulerType.linear_schedule_with_warmup: get_linear_schedule_with_warmup
}
