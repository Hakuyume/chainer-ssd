import cv2
import os
import xml.etree.ElementTree as ET

from rect import Rect


class VOC:

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

    def __init__(self, root, sets, difficult=True):
        self.root = root

        self.images = list()
        for year, name in sets:
            root = os.path.join(self.root, 'VOC' + year)
            for line in open(
                    os.path.join(root, 'ImageSets', 'Main', name + '.txt')):
                self.images.append((root, line.strip()))

        self.difficult = difficult

    def __len__(self):
        return len(self.images)

    def name(self, i):
        return self.images[i][1]

    def image(self, i):
        return cv2.imread(
            os.path.join(
                self.images[i][0], 'JPEGImages', self.images[i][1] + '.jpg'),
            cv2.IMREAD_COLOR)

    def annotations(self, i):
        annotations = list()

        tree = ET.parse(os.path.join(
            self.images[i][0], 'Annotations', self.images[i][1] + '.xml'))
        for child in tree.getroot():
            if not child.tag == 'object':
                continue

            if not self.difficult and bool(int(child.find('difficult').text)):
                continue

            bndbox = child.find('bndbox')
            rect = Rect.LTRB(*(
                float(bndbox.find(t).text)
                for t in ('xmin', 'ymin', 'xmax', 'ymax')))
            cls = self.names.index(child.find('name').text)

            annotations.append((rect, cls))

        return annotations
