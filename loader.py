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

    def get_example(self, i):
        image = self.dataset.image(i)
        boxes, classes = self.dataset.annotation(i)
        boxes = np.array(boxes)
        classes = np.array(classes)

        h, w, _ = image.shape
        if random.randrange(2):
            image = image[:, ::-1]
            boxes[:, 0::2] = w - boxes[:, 2::-2]

        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        image -= self.mean
        image = image.transpose(2, 0, 1)
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        loc, conf = self.encoder.encode(boxes, classes)

        return image, loc, conf
