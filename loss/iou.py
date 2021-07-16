from jittor import nn
from jittor import Module


class IoULoss(Module):
    def __init__(self, smooth=1):
        self.smooth = smooth

    def execute(self, input, target):     
        '''        
        :param input:  [batch_size,c,h,w]
        :param target: [batch_size,h,w]
        :return: loss
        '''
        input = nn.softmax(input, dim = 1)
        
        c_dim = input.shape[1]
        input = input.transpose((0, 2, 3, 1)).reshape((-1, c_dim))
        
        target = target.reshape((-1, ))
        target = target.broadcast(input, [1])
        target = target.index(1) == target

        smooth = 1.0
        iflat = input.view((- 1)).float()
        tflat = target.view((- 1)).float()
        intersection = (iflat * tflat).sum()
        return (1 - ((intersection + smooth) / ((iflat.sum() + tflat.sum() - intersection) + smooth)))
