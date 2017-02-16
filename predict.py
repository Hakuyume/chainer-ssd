#! /usr/bin/env python3

import argparse
import cv2
import numpy as np

from chainer import serializers

import config
from ssd import SSD300
from multibox import MultiBoxEncoder
from rect import Rect
import voc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    model = SSD300(n_class=20, aspect_ratios=config.aspect_ratios)
    serializers.load_npz(args.model, model)

    multibox_encoder = MultiBoxEncoder(
        grids=model.grids,
        aspect_ratios=model.aspect_ratios,
        variance=config.variance)

    src = cv2.imread(args.image, cv2.IMREAD_COLOR)

    x = cv2.resize(src, (model.insize, model.insize)).astype(np.float32)
    x -= config.mean
    x = x.transpose(2, 0, 1)
    x = x[np.newaxis]

    loc, conf = model(x)
    boxes, conf = multibox_encoder.decode(loc.data[0], conf.data[0])
    conf = conf[:, 1:]

    img = src.copy()
    selected = set()
    for i in conf.max(axis=1).argsort()[::-1]:
        box = Rect.LTRB(*boxes[i]).scale(*img.shape[:2][::-1])
        if len(selected) > 0:
            iou = max(box.iou(s) for s in selected)
            if iou > 0.45:
                continue
        selected.add(box)

        if conf[i].max() < 0.9:
            break

        cv2.rectangle(
            img,
            (int(box.left), int(box.top)),
            (int(box.right), int(box.bottom)),
            (0, 0, 255),
            3)

        name = voc.names[conf[i].argmax()]
        (w, h), b = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        cv2.rectangle(
            img,
            (int(box.left), int(box.top)),
            (int(box.left + w), int(box.top + h + b)),
            (0, 0, 255),
            -1)
        cv2.putText(
            img,
            name,
            (int(box.left), int(box.top + h)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255))

    while True:
        cv2.imshow('result', img)
        if cv2.waitKey() == ord('q'):
            break
