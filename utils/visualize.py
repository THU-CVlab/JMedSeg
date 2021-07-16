from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os
from utils import isImageFile


plt.switch_backend('agg')


def paintContour(model, paint_loader, mask_flag, args):
    pbar = tqdm(total = len(paint_loader), desc=f"paint coutour")
    if mask_flag: # if groundtruth mask is available
        for idx, (img_file_name, img, img_, mask) in enumerate(paint_loader):
            img_file_name = img_file_name[0]
            img = np.array(img[0])     # 转换为RGB
            
            pred = model(img_)
            pred = np.argmax(pred, axis=1)
            pred = np.array(pred).astype(np.uint8)  # pred.shape = (1, 512, 512)
            contours, hierarchy = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # findContours shape = (512, 512)
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    img[y, x, :] = [255, 255, 0]

            mask = np.array(mask).astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # findContours shape = (512, 512)
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    img[y, x, :] = [0, 0, 255]

            save_path = os.path.join('test_performance', f"{args.model}-{args.optimizer}-{args.loss}-{args.epochs}-{args.mode}-stn{str(args.stn)}-pretrain{str(args.pretrain)}-aug{str(args.aug)}")
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            dst = os.path.join(save_path, img_file_name.split('/')[-1]).replace('.jpg', '.png')
            cv2.imwrite(dst, img)
            # print ('Test in instance {}'.format(idx))
            pbar.update()

    else: # elsewise, the groundtruth mask does not exist
        for idx, (img_file_name, img, img_) in enumerate(paint_loader):
            img_file_name = img_file_name[0]
            img = np.array(img[0])     # 转换为RGB

            pred = model(img_)
            pred = np.argmax(pred, axis=1)
            pred = np.array(pred).astype(np.uint8)
            contours, hierarchy = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    img[y, x, :] = [255, 255, 0]

            root_path = os.path.join('test_performance', f"{args.model}-{args.optimizer}-{args.loss}-{args.epochs}-{args.mode}-stn{str(args.stn)}-pretrain{str(args.pretrain)}-aug{str(args.aug)}")
            
            if not os.path.exists(root_path):
                os.makedirs(root_path)
                
            sub_files = img_file_name.split('/')
            for sub_file in sub_files:
                if isImageFile(sub_file):
                    break
                root_path = root_path + '/' + sub_file
                if not os.path.exists(root_path):
                    os.mkdir(root_path)
            dst = os.path.join(root_path, img_file_name.replace('.jpg', '.png'))
            cv2.imwrite(dst, img)
            pbar.update()
    pbar.close()


def paintResult(epoch_index_list, epoch_loss_list, epoch_miou_list, epoch_mdsc_list, args):
    plt.figure(figsize=(20,30), dpi=80, facecolor = "white")
    ax1 = plt.subplot(3,1,1)
    ax1.set_title("Loss")
    plt.plot(epoch_index_list, epoch_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    ax2 = plt.subplot(3,1,2)
    ax2.set_title("mIoU")
    plt.plot(epoch_index_list, epoch_miou_list)
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    ax3 = plt.subplot(3,1,3)
    ax3.set_title("mDSC")
    plt.plot(epoch_index_list, epoch_mdsc_list)
    plt.xlabel('Epochs')
    plt.ylabel('mDSC')
    plt.legend()
    plt.savefig(f'./result/{args.model}-{args.optimizer}-{args.loss}-{args.epochs}.png')