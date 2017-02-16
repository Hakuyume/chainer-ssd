import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET


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


class VOCDataset:

    def __init__(self, root, sets):
        self.root = root

        self.images = list()
        for year, name in sets:
            root = os.path.join(self.root, 'VOC' + year)
            for line in open(
                    os.path.join(root, 'ImageSets', 'Main', name + '.txt')):
                self.images.append((root, line.strip()))

    def __len__(self):
        return len(self.images)

    def name(self, i):
        return self.images[i][1]

    def image(self, i):
        return cv2.imread(
            os.path.join(
                self.images[i][0], 'JPEGImages', self.images[i][1] + '.jpg'),
            cv2.IMREAD_COLOR)

    def annotation(self, i):
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
        return boxes, classes
