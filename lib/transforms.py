import cv2
import numpy as np
import random

from lib import matrix_iou


def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes) == 0:
        return image, boxes, labels

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
            return image, boxes, labels

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

            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)
            boxes = boxes[mask].copy()
            labels = labels[mask]

            boxes[:, :2] = np.maximum(boxes[:, :2], roi[:2])
            boxes[:, :2] -= roi[:2]
            boxes[:, 2:] = np.minimum(boxes[:, 2:], roi[2:])
            boxes[:, 2:] -= roi[:2]

            return image, boxes, labels


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, fill):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape
    ratio = random.uniform(1, 4)
    left = random.randint(0, int(width * ratio) - width)
    top = random.randint(0, int(height * ratio) - height)

    expand_image = np.empty(
        (int(height * ratio), int(width * ratio), depth),
        dtype=image.dtype)
    expand_image[:] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    boxes = boxes.copy()
    boxes[:, :2] += (left, top)
    boxes[:, 2:] += (left, top)

    return image, boxes


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc_for_test(image, insize, mean):
    image = cv2.resize(image, (insize, insize))
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)


def preproc_for_train(image, boxes, labels, insize, mean):
    if len(boxes) == 0:
        boxes = np.empty((0, 4))

    image, boxes, labels = _crop(image, boxes, labels)
    image = _distort(image)
    image, boxes = _expand(image, boxes, mean)
    image, boxes = _mirror(image, boxes)

    height, width, _ = image.shape
    image = preproc_for_test(image, insize, mean)
    boxes = boxes.copy()
    boxes[:, 0::2] /= width
    boxes[:, 1::2] /= height

    return image, boxes, labels
