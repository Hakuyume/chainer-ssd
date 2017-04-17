#! /usr/bin/env python3

import argparse
import cv2
import numpy as np

from chainer import serializers

import config
from lib import MultiBoxEncoder
from lib import SSD300
from lib import VOCDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    model = SSD300(20)
    serializers.load_npz(args.model, model)

    multibox_encoder = MultiBoxEncoder(
        model=model,
        steps=config.steps,
        sizes=config.sizes,
        variance=config.variance)

    src = cv2.imread(args.image, cv2.IMREAD_COLOR)

    x = cv2.resize(src, (model.insize, model.insize)).astype(np.float32)
    x -= config.mean
    x = x.transpose(2, 0, 1)
    x = x[np.newaxis]

    loc, conf = model(x)
    results = multibox_encoder.decode(loc.data[0], conf.data[0], 0.45, 0.01)

    img = src.copy()
    for box, label, score in results:
        box = np.array(box)
        box[:2] *= img.shape[1::-1]
        box[2:] *= img.shape[1::-1]
        box = box.astype(int)

        print(label + 1, score, *box)

        if score < 0.6:
            continue

        cv2.rectangle(
            img,
            (box[0], box[1]), (box[2], box[3]),
            (0, 0, 255),
            3)

        name = VOCDataset.labels[label]
        (w, h), b = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        cv2.rectangle(
            img,
            (box[0], box[1]), (box[0] + w, box[1] + h + b),
            (0, 0, 255),
            -1)
        cv2.putText(
            img,
            name,
            (box[0], box[1] + h),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255))

    print('(press \'q\' to exit)')
    while True:
        cv2.imshow('result', img)
        if cv2.waitKey() == ord('q'):
            break
