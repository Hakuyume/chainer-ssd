#! /usr/bin/env python3

import argparse
import os

import chainer
from chainer import iterators
from chainer import serializers

from lib import MultiBoxEncoder
from lib import preproc_for_test
from lib import SSD300
from lib import VOCDataset


class TestDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.insize = model.insize
        self.mean = model.mean

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        name = self.dataset.name(i)
        image = self.dataset.image(i)
        height, width, _ = image.shape
        image = preproc_for_test(image, self.insize, self.mean)
        return name, image, (width, height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='VOCdevkit')
    parser.add_argument('--output', default='result')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('model')
    parser.add_argument('test')
    args = parser.parse_args()

    model = SSD300(20)
    serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    multibox_encoder = MultiBoxEncoder(model)

    year, subset = args.test.split('-')
    dataset = TestDataset(VOCDataset(args.root, year, subset), model)

    iterator = iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    os.makedirs(args.output, exist_ok=True)
    files = [
        open(
            os.path.join(
                args.output, 'comp4_det_test_{:s}.txt'.format(label)),
            mode='w')
        for label in VOCDataset.labels]

    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        x = model.xp.array([image for _, image, _ in batch])
        loc, conf = model(x)
        loc = chainer.cuda.to_cpu(loc.data)
        conf = chainer.cuda.to_cpu(conf.data)

        for (name, _, size), loc, conf in zip(batch, loc, conf):
            print(name)

            boxes, labels, scores = multibox_encoder.decode(
                loc, conf, 0.45, 0.01)
            for box, label, score in zip(boxes, labels, scores):
                box[:2] *= size
                box[2:] *= size

                print(name, score, *box, file=files[label])

    for f in files:
        f.close()
