import jittor as jt
import numpy as np
from advance import *
import matplotlib.pyplot as plt
import argparse
import matplotlib.pyplot as plt
from tqdm import trange
from utils import get_model, modelSet, dataset_choices
import argparse 

plt.switch_backend('agg')

# CUDA_VISIBLE_DEVICES=0 log_silent=1 python3.7 run_ssl.py --model deeplab --layer aspp --channel 256 --dataset pancreas --save checkpoints/deeplab-ssl.pkl -e 50 --lr 5e-6

if __name__ == '__main__':
    jt.flags.use_cuda = int(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='unet', type=str, choices=modelSet, help='choose a model network')
    parser.add_argument('--dataset', type=str, choices=dataset_choices, required=True, help='select a dataset')
    parser.add_argument('--save', default='checkpoints/ssl.pkl', type=str, help='model weights save path')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of training epochs', dest='epochs')
    parser.add_argument('-c', '--class-num', type=int, default=2, help='class number', dest='class_num')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='training batch size', dest='batch_size')
    parser.add_argument('--channel', dest='embedding_channel', type=int, default=512, help='number of channels of embedded feature maps')
    parser.add_argument('--layer', type=str, default='down4', help='layer to extract features from')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--pretrain', action='store_true')
    args = parser.parse_args()

    model = get_model(args)
    train_loader = retrieve_aug_data(args, 'train', augmentation)

    learner = MoCoLearner(
        model=model,
        layer=args.layer,
        loader=train_loader,
        embedding_channel=args.embedding_channel,
        project_dim=128,
        lr=args.lr
    )

    loss_min = 1e4
    losses = []
    with open('./log/ssl.txt', 'w') as f:
        # bar = trange(args.epochs)
        for epoch in range(args.epochs):
            loss = learner.train()
            # bar.set_description('epoch[%02d] loss:[%.6f\n]' % (epoch + 1, loss))
            print('epoch[%02d] loss:[%.6f\n]' % (epoch + 1, loss))
            f.write('epoch[%02d] loss:[%.6f\n]' % (epoch + 1, loss))
            if loss < loss_min:
                model.save(args.save)
            losses.append(loss)
    np.savetxt('./log/ssl_loss.txt', loss)
    plt.plot(losses)
    plt.savefig('./result/ssl_losses.png')