import cv2
import numpy as np


class Evaluator(object):
    def __init__(self, num_class = 2):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def dice(self):
        dice = 2.0 * np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + 
            np.sum(self.confusion_matrix, axis=0))
        dice = np.nanmean(dice)
        return dice

    def recall(self):
        tp, fn, fp = self.confusion_matrix[0][0], self.confusion_matrix[0][1], self.confusion_matrix[1][0]
        precision = tp / (tp + fn)
        return precision

    def precision(self):
        tp, fn, fp = self.confusion_matrix[0][0], self.confusion_matrix[0][1], self.confusion_matrix[1][0]
        precision = tp / (tp + fp)
        return precision

    def accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def class_accuracy(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def iou(self):
        iou= np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + 
            np.sum(self.confusion_matrix, axis=0)-
            np.diag(self.confusion_matrix))
        iou = np.nanmean(iou)
        return iou

    def fwiou(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou

    def _generate_matrix(self, truth, pred):
        mask = (truth >= 0) & (truth < self.num_class)
        label = self.num_class * truth[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, truth, pred):
        assert truth.shape == pred.shape
        self.confusion_matrix += self._generate_matrix(truth, pred)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def calDistance(self, seg_A, seg_B, dx=1.0, k=1):
        # Extract the label k from the segmentation maps to generate binary maps
        seg_A = np.array(seg_A)
        seg_B = np.array(seg_B)

        seg_A = (seg_A == k)
        seg_B = (seg_B == k)

        table_md = []
        table_hd = []

        Z, Y, X = seg_A.shape

        for z in range(Z):
            # Binary mask at this slice
            slice_A = seg_A[z, :, :].astype(np.uint8)
            slice_B = seg_B[z, :, :].astype(np.uint8)

            # The distance is defined only when both contours exist on this slice
            if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
                # Find contours and retrieve all the points
                contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                pts_A = contours[0]
                for i in range(1, len(contours)):
                    pts_A = np.vstack((pts_A, contours[i]))

                contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                pts_B = contours[0]
                for i in range(1, len(contours)):
                    pts_B = np.vstack((pts_B, contours[i]))

                # Distance matrix between point sets
                M = np.zeros((len(pts_A), len(pts_B)))
                for i in range(len(pts_A)):
                    for j in range(len(pts_B)):
                        M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

                # Mean distance and hausdorff distance
                md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
                hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
                table_md += [md]
                table_hd += [hd]

        # Return the mean distance and Hausdorff distance across 2D slices
        mean_md = np.mean(table_md) if table_md else None
        mean_hd = np.mean(table_hd) if table_hd else None
        return mean_md, mean_hd