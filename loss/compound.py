# from loss.crossentropy import CrossEntropyLoss
from loss.dice import DiceLoss
from loss.focal import FocalLoss
from loss.iou import IoULoss
import jittor.nn as nn
import jittor as jt


# target loss functions
criterion_dict = {
    'ce': nn.CrossEntropyLoss(), 
    'iou':IoULoss(), 
    'dice':DiceLoss(), 
    'focal':FocalLoss()
}


def get_criterion(args):
    criterion = criterion_dict[args.loss]
    if args.loss == 'ce':
        weight = jt.Var([1.0 - args.bce_weight, args.bce_weight])
        criterion = nn.CrossEntropyLoss(weight=weight)
    return criterion


loss_help_msg = """
    Choose from 'ce', 'iou', 'dice', 'focal', 
    if CE loss is selected, you should use a `weight` parameter
"""


def get_criterion_and_ratio(args):
    criterions = [] 
    ratio = [] 

    arg_loss = args.loss.split('_')
    if len(arg_loss) == 1:
        assert arg_loss[0] in criterion_dict.keys(), loss_help_msg
        criterions.append(criterion_dict[arg_loss[0]])
        ratio.append(1)
    else:
        assert len(arg_loss) % 2 == 0, loss_help_msg
        for i in range(0, len(arg_loss), 2):
            assert arg_loss[i] in criterion_dict.keys(), loss_help_msg
            criterions.append(criterion_dict[arg_loss[i]])
            ratio.append(float(arg_loss[i+1]))
    return criterions, ratio

