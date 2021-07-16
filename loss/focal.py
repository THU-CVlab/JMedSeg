import jittor as jt
from jittor import Module


class FocalLoss(Module):
    def __init__(self, gamma=2, alpha=0.5):
        self.gamma = gamma
        self.alpha = alpha
    
    def cross_entropy_loss(self, input, target):
        if len(input.shape) == 4:
            c_dim = input.shape[1]
            input = input.transpose((0, 2, 3, 1))
            input = input.reshape((-1, c_dim))
        target = target.reshape((-1, ))
        target = target.broadcast(input, [1])
        target = target.index(1) == target
        
        input = input - input.max([1], keepdims=True)
        loss = input.exp().sum(1).log()
        loss = loss - (input*target).sum(1)
        
        return loss

    def execute(self, input, target):
        '''
        :param input: [batch_size,c,h,w]
        :param target: [batch_size,h,w]
        :return: loss
        '''
        logpt = -1 * self.cross_entropy_loss(input, target)
        pt = jt.exp(logpt)
        loss = (- self.alpha * ((1 - pt) ** self.gamma)) * logpt
        return loss.mean()        
