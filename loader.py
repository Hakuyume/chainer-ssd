import cv2
import numpy as np
import random

import chainer


class SSDLoader(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, size, mean, encoder):
        super().__init__()

        self.dataset = dataset
        self.size = size
        self.mean = mean
        self.encoder = encoder

    def __len__(self):
        return len(self.dataset)

    def augment(self, image, boxes, classes):
        if len(boxes) == 0:
            raise ValueError('boxes is empty')

        h, w, _ = image.shape

        mode = random.randrange(2)
        while True:
            if mode == 0:
                patch = (0, 0, w, h)
            elif mode == 1:
                size = random.uniform(0.1, 1) * w * h
                aspect = random.uniform(
                    max(0.5, size / (w * w)),
                    min(2, (h * h) / size))
                patch_w = np.sqrt(size / aspect)
                patch_h = np.sqrt(size * aspect)
                patch_l = random.uniform(0, w - patch_w)
                patch_t = random.uniform(0, h - patch_h)
                patch = (
                    patch_l, patch_t, patch_l + patch_w, patch_t + patch_h)

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(
                (patch[:2] <= centers).all(axis=1),
                (patch[2:] > centers).all(axis=1))
            if mask.any():
                break

        image = image[
            int(patch[1]):int(patch[3]),
            int(patch[0]):int(patch[2])]
        boxes = boxes[mask]
        classes = classes[mask]

        boxes = np.hstack((
            np.maximum(boxes[:, :2], patch[:2]) - patch[:2],
            np.minimum(boxes[:, 2:], patch[2:]) - patch[:2]))

        if random.random() < 0.5:
            image = image[:, ::-1]
            boxes[:, ::2] = patch[2] - patch[0] - boxes[:, ::2]

        return image, boxes, classes

    def get_example(self, i):
        image, boxes, classes = self.dataset[i]
        image, boxes, classes = self.augment(image, boxes, classes)

        h, w, _ = image.shape
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        image -= self.mean
        image = image.transpose(2, 0, 1)
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        loc, conf = self.encoder.encode(boxes, classes)

        return image, loc, conf
