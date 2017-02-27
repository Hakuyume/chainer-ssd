#! /usr/bin/env python3

import argparse
import cv2
import numpy as np
import os

import chainer
from chainer import cuda
from chainer import serializers

import config
from ssd import SSD300
from multibox import MultiBoxEncoder
from voc import VOC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='VOCdevkit')
    parser.add_argument('--output', default='.')
    parser.add_argument('--test', action='append')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('model')
    args = parser.parse_args()

    model = SSD300(n_classes=20, aspect_ratios=config.aspect_ratios)
    serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    multibox_encoder = MultiBoxEncoder(
        grids=model.grids,
        steps=config.steps,
        sizes=config.sizes,
        aspect_ratios=model.aspect_ratios,
        variance=config.variance)

    def dump_result(name, size, loc, conf):
        print(name)

        if xp is not np:
            loc = xp.asnumpy(loc)
            conf = xp.asnumpy(conf)

        boxes, conf = multibox_encoder.decode(loc, conf)
        nms = multibox_encoder.non_maximum_suppression(boxes, conf, 0.45, 0.01)
        for box, cls, score in nms:
            box *= size
            filename = os.path.join(
                args.output, 'comp4_det_test_{:s}.txt'.format(VOC.names[cls]))
            with open(filename, mode='a') as f:
                print(
                    name, score, box.left, box.top, box.right, box.bottom,
                    file=f)

    dataset = VOC(args.root, [t.split('-') for t in args.test])

    info = list()
    batch = list()
    for i in range(len(dataset)):
        src = dataset.image(i)
        info.append((dataset.name(i), src.shape[1::-1]))
        x = cv2.resize(src, (model.insize, model.insize)).astype(np.float32)
        x -= config.mean
        x = x.transpose(2, 0, 1)
        batch.append(x)

        if len(batch) == args.batchsize:
            loc, conf = model(chainer.Variable(xp.array(batch), volatile=True))
            for i, (name, size) in enumerate(info):
                dump_result(name, size, loc.data[i], conf.data[i])
            info = list()
            batch = list()

    if len(batch) > 0:
        loc, conf = model(chainer.Variable(xp.array(batch), volatile=True))
        for i, (name, size) in enumerate(info):
            dump_result(name, size, loc.data[i], conf.data[i])
