import cv2
import numpy as np
import random

from rect import Rect, matrix_iou

import chainer


def crop(image, boxes, classes):
    height, width, _ = image.shape

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, classes

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            w = random.randrange(int(0.3 * width), width)
            h = random.randrange(int(0.3 * height), height)

            if h / w < 0.5 or 2 < h / w:
                continue

            rect = Rect.LTWH(
                random.randrange(width - w),
                random.randrange(height - h),
                w,  h)

            iou = matrix_iou(boxes, np.array((rect,)))
            if iou.min() < min_iou and max_iou < iou.max():
                continue

            image = image[rect.top:rect.bottom, rect.left:rect.right]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(
                (rect.left, rect.top) < centers,
                centers < (rect.right, rect.bottom)).all(axis=1)
            boxes = boxes[mask].copy()
            classes = classes[mask]

            boxes[:, :2] = np.maximum(boxes[:, :2], (rect.left, rect.top))
            boxes[:, :2] -= (rect.left, rect.top)
            boxes[:, 2:] = np.minimum(boxes[:, 2:], (rect.right, rect.bottom))
            boxes[:, 2:] -= (rect.left, rect.top)

            return image, boxes, classes


def distort(image, boxes, classes):
    def convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image, boxes, classes


def expand(image, boxes, classes, mean):
    if random.randrange(2):
        return image, boxes, classes

    height, width, depth = image.shape
    ratio = random.uniform(1, 4)
    left = random.randint(0, int(width * ratio) - width)
    top = random.randint(0, int(height * ratio) - height)

    expand_image = np.empty(
        (int(height * ratio), int(width * ratio), depth),
        dtype=image.dtype)
    expand_image[:, :] = mean
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    boxes = boxes.copy()
    boxes[:, :2] += (left, top)
    boxes[:, 2:] += (left, top)

    return image, boxes, classes


def mirror(image, boxes, classes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes, classes


def augment(image, boxes, classes, mean):
    image, boxes, classes = crop(image, boxes, classes)
    image, boxes, classes = distort(image, boxes, classes)
    image, boxes, classes = expand(image, boxes, classes, mean)
    image, boxes, classes = mirror(image, boxes, classes)
    return image, boxes, classes


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

        try:
            boxes, classes = zip(*self.dataset.annotations(i))
            boxes = np.array(boxes)
            classes = np.array(classes)
            image, boxes, classes = augment(image, boxes, classes, self.mean)
        except ValueError:
            boxes = np.empty((0, 4), dtype=np.float32)
            classes = np.empty((0,), dtype=np.int32)

        h, w, _ = image.shape
        image = cv2.resize(
            image, (self.size, self.size)).astype(np.float32)
        image -= self.mean
        image = image.transpose(2, 0, 1)
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h

        loc, conf = self.encoder.encode(boxes, classes)

        return image, loc, conf
