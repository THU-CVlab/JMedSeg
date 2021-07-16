import os
import json
import numpy as np
import jittor as jt   
from jittor import nn
import argparse
from tqdm import tqdm
from model import summary
from loss import get_criterion, loss_help_msg
from utils import Evaluator, retrieve_data, dataset_choices, paintResult, paintContour, get_model, modelSet
from advance import STNWrapper, aug_for_unet


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='unet', type=str, choices=modelSet, help='choose the model')
parser.add_argument('--pretrain', action='store_true', help='whether to use pretrained weights')
parser.add_argument('--checkpoint', default='checkpoints/ssl.pkl', type=str, help='the location of the pretrained weights')
parser.add_argument('--dataset', type=str, choices=dataset_choices, required=True, help='choose a dataset')
parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict', 'debug'], required=True, help='select a mode')
parser.add_argument('--load', type=str, help='the location of the model weights for testing')
parser.add_argument('--aug', action='store_true', help='whether to use color augmentation')
parser.add_argument('--cuda', action='store_true', help='whether to use CUDA acceleration')
parser.add_argument('--stn', action='store_true', help='whether to use spatial transformer network')
parser.add_argument('-o', '--optimizer', default='Adam', type=str, choices=['Adam', 'SGD'], dest='optimizer', help='select an optimizer')
parser.add_argument('-e', '--epochs', type=int, default=20, dest='epochs', help='num of training epochs')
parser.add_argument('-b', '--batch-size', type=int, default=8, dest='batch_size', help='batch size for training')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, dest='lr', help='learning rate')
parser.add_argument('-c', '--class-num', type=int, default=2, dest='class_num', help='pixel-wise classes')
parser.add_argument('--loss', type=str, default='ce', help=loss_help_msg)
parser.add_argument('-w', dest='bce_weight', type=float, default=1.0, help='use this weight if BCE loss is selected; if w is given, then the weights for positive and negative classes will be w and 1.0 - w respectively')
parser.add_argument('-r', '--resultdir', dest='result_dir', type=str, help='test result output directory', default='result')
parser.add_argument('--poly', dest='poly_lr', action='store_true', help='whether to use polynomial learning rate scheduler')
args = parser.parse_args()

jt.flags.use_cuda = 1 if args.cuda else 0 
print('======='*10, '\n' , 'args:\n', str(args).replace('Namespace','\t').replace(", ", ",\n\t"), '\n' , '======='*10)


# learning rate scheduler
def poly_lr_scheduler(opt, init_lr, iter, epoch, max_iter, max_epoch):
    new_lr = init_lr * (1 - float(epoch * max_iter + iter) / (max_epoch * max_iter)) ** 0.9
    opt.lr = new_lr


# compound loss function calculation
criterion = get_criterion(args)

def cal_loss(input, target):
    return criterion(input, target)


# train function
def train(model, train_loader, optimizer, epoch, init_lr):
    model.train()
    max_iter = len(train_loader)
    
    loss_list = []
    pbar = tqdm(total = max_iter, desc=f"epoch {epoch} train")
    for idx, (imgs, target) in enumerate(train_loader):
        if args.poly_lr:
            poly_lr_scheduler(optimizer, init_lr, idx, epoch, max_iter, 50)
        imgs = imgs.float32()
        pred = model(imgs)
        loss = cal_loss(pred, target)
        optimizer.step (loss)
        loss_list.append(loss.data[0])
        pbar.set_postfix({'loss': loss_list[-1]})
        pbar.update(1)
        del pred, loss
    pbar.close()
    return np.mean(loss_list)


# validate function
def val(model, val_loader, epoch, evaluator, best_miou):
    model.eval()
    evaluator.reset()

    n_val = len(val_loader)
    pbar = tqdm(total = n_val, desc=f"epoch {epoch} valid")
    
    for _, (imgs, target) in enumerate(val_loader):
        imgs = imgs.float32()
        output = model(imgs)
        pred = output.data
        target = target.data
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
        pbar.update(1)

    pbar.close()
    Acc = evaluator.accuracy()
    Acc_class = evaluator.class_accuracy()
    mIoU = evaluator.iou()
    FWIoU = evaluator.fwiou()
    dice = evaluator.dice()

    if (mIoU > best_miou):
        best_miou = mIoU
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        if not os.path.exists(f'./checkpoints/{args.model}'):
            os.mkdir(f'./checkpoints/{args.model}')
        model_path = f'./checkpoints/{args.model}/{args.model}-{args.optimizer}-{args.loss}-stn{str(args.stn)}-pretrain{str(args.pretrain)}-aug{str(args.aug)}.pkl'
        model.save(model_path)
    print ('Testing result of epoch {}: miou = {} Acc = {} Acc_class = {} FWIoU = {} Best Miou = {} DSC = {}'.format(epoch, mIoU, Acc, Acc_class, FWIoU, best_miou, dice))
    return best_miou, mIoU, dice


# test function
def test(model, test_loader, evaluator):
    model.eval()
    evaluator.reset()

    n_test = len(test_loader)

    pbar = tqdm(total = n_test, desc=f"test")
    for _, (imgs, target) in enumerate(test_loader):
        output = model(imgs)
        pred = output.data
        target = target.data
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
        pbar.update(1)
    pbar.close()

    Acc = evaluator.accuracy()
    Acc_class = evaluator.class_accuracy()
    mIoU = evaluator.iou()
    FWIoU = evaluator.fwiou()
    dice = evaluator.dice()
    recall = evaluator.recall()
    precision = evaluator.precision()

    result = {
        "mDice": dice,
        'mIoU': mIoU,
        'mFWIoU': FWIoU,
        "mPrecision": precision,
        "mRecall": recall,
        "mAcc": Acc,
        "mAcc_class": Acc_class,
    }
    print ('Testing result of {}: mIoU = {} Acc = {} Acc_class = {} FWIoU = {} mDice = {} mPrecision = {} mRecall = {}'.format(
            args.model, mIoU, Acc, Acc_class, FWIoU, dice, precision, recall
        ))

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    json.dump(result, open(os.path.join(args.result_dir, f"{args.model}-{args.optimizer}-{args.loss}-{args.epochs}-{args.mode}-stn{str(args.stn)}-pretrain{str(args.pretrain)}-aug{str(args.aug)}.json"),"w"), indent=2, ensure_ascii=False)
    return result


# model network
model = get_model(args)

# spatial transformer localization network
if args.stn:
    model = STNWrapper(model)

# Data augmentation
aug = aug_for_unet if args.aug else None

# Learning hyper-parameters
batch_size = args.batch_size
lr = args.lr
if args.optimizer == 'SGD':
    optimizer = nn.SGD(model.parameters(), lr, momentum = 0.9, weight_decay = 1e-4)
else:
    optimizer = nn.Adam(model.parameters(), lr, betas = (0.9,0.999))
    
epochs = args.epochs
best_miou = 0.0
best_mdsc = 0.0
epoch_index_list = list(range(epochs))
epoch_loss_list = []
epoch_miou_list = []
epoch_mdsc_list = []
evaluator = Evaluator(num_class = args.class_num)


load_path = f'./checkpoints/{args.model}/{args.model}-{args.optimizer}-{args.loss}-stn{str(args.stn)}-pretrain{str(args.pretrain)}-aug{str(args.aug)}.pkl' if args.load is None else args.load

# dataset and main loop
if args.mode == 'train':
    train_loader = retrieve_data(args, 'train', aug)
    test_loader = retrieve_data(args, 'test', aug=None)
    val_loader = retrieve_data(args, 'val', aug=None)
    paint_loader = retrieve_data(args, 'test', aug=None, paint=True)
    for epoch in range(epochs):
        epoch_loss = train(model, train_loader, optimizer, epoch, lr)
        best_miou, mIoU, mdsc = val(model, val_loader, epoch, evaluator, best_miou)
        epoch_loss_list.append(epoch_loss)
        epoch_miou_list.append(mIoU)
        epoch_mdsc_list.append(mdsc)
    paintResult(epoch_index_list, epoch_loss_list, epoch_miou_list, epoch_mdsc_list, args)
    model.load(load_path)
    result = test(model, test_loader, evaluator)
    paintContour(model, paint_loader, True, args)
elif args.mode == 'test':
    model.load(load_path)
    test_loader = retrieve_data(args, 'test', aug=None)
    paint_loader = retrieve_data(args, 'test', aug=None, paint=True)
    result = test(model, test_loader, evaluator)
    paintContour(model, paint_loader, True, args)
elif args.mode == 'predict':
    model.load(load_path)
    paint_loader = retrieve_data(args, 'test', aug=None, paint=True, mask=False)
    paintContour(model, paint_loader, False, args)
elif args.mode == 'debug':
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print(y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    summary(model, input_size=(3, 512, 512), device='cuda')
