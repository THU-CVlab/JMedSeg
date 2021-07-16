# Created by Yuanbiao Wang
# Implements a simple contrastive learning pretrain learner

# MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
# https://github.com/facebookresearch/moco

import jittor as jt
import jittor.nn as nn
from advance.ssl_utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.switch_backend('agg')

class MoCo(nn.Module):
    def __init__(self, encoder, embedding_channel, projection_dim, K=1024, T=0.07, dim=128):
        super(MoCo, self).__init__()
        self.K = K
        self.T = T
        self.encoder = encoder
        self.project = Projection(embedding_channel, projection_dim)
        self.queue = jt.randn(dim, K)
        self.queue = jt.misc.normalize(self.queue, dim=0)
        self.ptr = 0
        
    def _dequeue_and_enqueue(self, keys):
        with jt.no_grad():
            batch_size = keys.shape[0]
            left_space = self.K - self.ptr
            key_size = min(batch_size, left_space)
            keys = keys[:key_size]
            self.queue[:, self.ptr : self.ptr + key_size] = keys.transpose()
            self.ptr = (self.ptr + key_size) % self.K 
        
    def execute(self, im_q, im_k):
        q = self.encoder(im_q)
        q = self.project(q)
        q = jt.misc.normalize(q, dim=1)                         # im_q feature vector
        k = self.encoder(im_k)
        k = self.project(k)
        k = jt.misc.normalize(k, dim=1)                         # im_k feature vector
        l_pos = (q * k).sum(dim=1).unsqueeze(-1)                # similarity of two feature vectors
        l_neg = jt.matmul(q, self.queue.clone().detach())       # discrepancy of two feature vectors, queue is the buffer
        logits = jt.contrib.concat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = jt.zeros(logits.shape[0], dtype=jt.int)
        self._dequeue_and_enqueue(k)                            # add im_k feature vector into the buffer
        return logits, labels
        # https://blog.csdn.net/yyhaohaoxuexi/article/details/113824125 博客解读
        # 这里的变量logits的意义我也查了一下：是未进入softmax的概率，crossentropy会自动做一个log softmax
        # 这段代码根据注释即可理解：l_pos表示正样本的得分，l_neg表示所有负样本的得分，logits表示将正样本和负样本在列上cat起来之后的值。
        # 值得关注的是，labels的数值，是根据logits.shape[0]的大小生成的一组zero。也就是大小为batch_size的一组0。
        # 这里直接对输出的logits和生成的````labels```计算交叉熵，然后就是模型的loss。这里就是让我不是很理解的地方。先将疑惑埋在心里～


class OutputHiddenLayer(nn.Module):
    def __init__(self, net, layer=(-2)):
        super().__init__()
        self.net = net
        self.layer = layer
        self.hidden = None
        self._register_hook()

    def _find_layer(self):
        if (type(self.layer) == str):
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif (type(self.layer) == int):
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _register_hook(self):
        def hook(_, __, output):
            self.hidden = output
        layer = self._find_layer()
        assert (layer is not None)
        handle = layer.register_forward_hook(hook)

    def execute(self, x):
        if (self.layer == (- 1)):
            return self.net(x)
        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert (hidden is not None)
        return hidden


class Projection(nn.Module):
    def __init__(self, input_channel, project_dim):
        super(Projection, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=2)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_channel * 4, project_dim)
        )
    
    def execute(self, x):
        y = self.pool(x)
        y = y.view(x.size(0), -1)
        y = self.fc(y)
        return y
    
        
class MoCoLearner():
    def __init__(self, model, layer, loader, embedding_channel=1024, project_dim=128, lr=1e-5):
        super(MoCoLearner, self).__init__()
        encoder = OutputHiddenLayer(model, layer)
        self.co = MoCo(encoder, embedding_channel, project_dim)
        self.loader = loader
        self.criterion = nn.CrossEntropyLoss()
        self.optim = jt.optim.Adam(model.parameters(), lr=lr)
    
    def update(self, query, key):
        output, target = self.co(query, key)
        loss = self.criterion(output, target)
        self.optim.step(loss)
        return loss.item()
    
    def train(self):
        loss_mean = 0.0
        total = 0
        bar = tqdm(self.loader, desc='loss')
        for i, (query, key, _) in enumerate(bar):
            loss = self.update(query, key)
            bar.set_description('loss: [%.6f]' % loss)
            bar.update()
            loss_mean += loss * query.shape[0]
            total += query.shape[0]
        loss_mean /= total
        return loss_mean
    