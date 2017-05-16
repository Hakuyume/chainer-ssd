import cv2
import numpy as np
import random

from lib import matrix_iou


def _crop(image, boxes, labels):
    if len(boxes) == 0:
        return image, boxes, labels

    constraints = (
        (0.1, None),
        (0.3, None),
        (0.5, None),
        (0.7, None),
        (0.9, None),
        (None, 0.1),
    )

    height, width, _ = image.shape

    rois = [np.array((0, 0, width, height))]
    for min_iou, max_iou in constraints:
        for _ in range(50):
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            scale = random.uniform(0.3, 1)
            aspect_ratio = random.uniform(
                max(1 / 2, scale * scale), min(2, 1 / (scale * scale)))
            w = int(width * scale * np.sqrt(aspect_ratio))
            h = int(height * scale / np.sqrt(aspect_ratio))

            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                rois.append(roi)
                break

    roi = random.choice(rois)

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
    def convert(image, alpha=1, beta=0):
        image = image.astype(float) * alpha + beta
        image[image < 0] = 0
        image[image > 255] = 255
        return image.astype(np.uint8)

    def brightness(image):
        if random.randrange(2):
            return convert(image, beta=random.uniform(-32, 32))
        else:
            return image

    def contrast(image):
        if random.randrange(2):
            return convert(image, alpha=random.uniform(0.5, 1.5))
        else:
            return image

    def saturation(image):
        if random.randrange(2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 1] = convert(
                image[:, :, 1], alpha=random.uniform(0.5, 1.5))
            return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            return image

    def hue(image):
        if random.randrange(2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 0] = (
                image[:, :, 0].astype(int) + random.randint(-18, 18)) % 180
            image = convert(image, alpha=random.uniform(0.5, 1.5))
            return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            return image

    image = brightness(image)

    if random.randrange(2):
        image = contrast(image)
        image = saturation(image)
        image = hue(image)
    else:
        image = saturation(image)
        image = hue(image)
        image = contrast(image)

    return image


def _expand(image, boxes, ratio, fill):
    height, width, depth = image.shape
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
    image = image[:, ::-1]
    boxes = boxes.copy()
    boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _resize(image, boxes, insize):
    inters = (
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    )
    inter = random.choice(inters)

    height, width, _ = image.shape
    image = cv2.resize(image, (insize, insize), interpolation=inter)
    boxes = boxes.copy()
    boxes[:, 0::2] *= insize / width
    boxes[:, 1::2] *= insize / height
    return image, boxes


def preproc_for_test(image, insize, mean):
    image = cv2.resize(image, (insize, insize))
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)


def preproc_for_train(image, boxes, labels, insize, mean):
    image = _distort(image)

    if random.randrange(2):
        ratio = random.uniform(1, 4)
        image, boxes = _expand(image, boxes, ratio, mean)

    image, boxes, labels = _crop(image, boxes, labels)
    image, boxes = _resize(image, boxes, insize)

    if random.randrange(2):
        image, boxes = _mirror(image, boxes)

    image = image.astype(np.float32)
    image -= mean
    image = image.transpose(2, 0, 1)
    boxes = boxes / insize

    return image, boxes, labels
