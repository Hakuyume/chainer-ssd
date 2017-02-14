import cv2
import numpy as np
import os
import random
import xml.etree.ElementTree as ET

import chainer


names = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)


class VOCDataset(chainer.dataset.DatasetMixin):

    def __init__(self, root, sets, size, encoder):
        self.root = root
        self.size = size
        self.encoder = encoder

        self.images = list()
        for year, name in sets:
            root = os.path.join(self.root, 'VOC' + year)
            for line in open(
                    os.path.join(root, 'ImageSets', 'Main', name + '.txt')):
                self.images.append((root, line.strip()))

    def __len__(self):
        return len(self.images)

    def augment(self, image, boxes, classes):
        h, w, _ = image.shape

        mode = random.randrange(2)
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
            patch = (patch_l, patch_t, patch_l + patch_w, patch_t + patch_h)

        image = image[
            int(patch[1]):int(patch[3]),
            int(patch[0]):int(patch[2])]

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = np.logical_and(
            (patch[:2] <= centers).all(axis=1),
            (patch[2:] > centers).all(axis=1))
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
        image = cv2.imread(
            os.path.join(
                self.images[i][0], 'JPEGImages', self.images[i][1] + '.jpg'),
            cv2.IMREAD_COLOR).astype(np.float32)

        boxes = list()
        classes = list()
        tree = ET.parse(os.path.join(
            self.images[i][0], 'Annotations', self.images[i][1] + '.xml'))
        for child in tree.getroot():
            if not child.tag == 'object':
                continue
            bndbox = child.find('bndbox')
            boxes.append(tuple(
                float(bndbox.find(t).text)
                for t in ('xmin', 'ymin', 'xmax', 'ymax')))
            classes.append(names.index(child.find('name').text))
        boxes = np.array(boxes)
        classes = np.array(classes)

        image, boxes, classes = self.augment(image, boxes, classes)

        h, w, _ = image.shape
        image = cv2.resize(image, (self.size, self.size))
        image -= (103.939, 116.779, 123.68)
        image = image.transpose(2, 0, 1)
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        loc, conf = self.encoder.encode(boxes, classes)

        return image, loc, conf
