from enum import Enum

from torch.nn import MSELoss


class LossType(Enum):
    mse = 'mse'


LOSSES_CLASSES = {
    LossType.mse: MSELoss()
}
