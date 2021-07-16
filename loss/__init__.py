from .iou import IoULoss
from .dice import DiceLoss
from .focal import FocalLoss
from .crossentropy import CrossEntropyLoss
from .compound import get_criterion_and_ratio, loss_help_msg, get_criterion