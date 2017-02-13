import cv2
import numpy as np
import os
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

    def get_example(self, i):
        x = cv2.imread(
            os.path.join(
                self.images[i][0], 'JPEGImages', self.images[i][1] + '.jpg'),
            cv2.IMREAD_COLOR)
        h, w, _ = x.shape
        x = cv2.resize(x, (self.size, self.size)).astype(np.float32)
        x -= (103.939, 116.779, 123.68)
        x = x.transpose(2, 0, 1)

        boxes = list()
        classes = list()
        tree = ET.parse(os.path.join(
            self.images[i][0], 'Annotations', self.images[i][1] + '.xml'))
        for child in tree.getroot():
            if not child.tag == 'object':
                continue
            bndbox = child.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            boxes.append((
                (xmin + xmax) / (2 * w),
                (ymin + ymax) / (2 * h),
                (xmax - xmin) / w,
                (ymax - ymin) / h))
            classes.append(names.index(child.find('name').text))
        loc, conf = self.encoder.encode(
            np.array(boxes), np.array(classes))

        return x, loc, conf
