#! /usr/bin/env python3

import argparse
import numpy as np

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from lib import CustomWeightDecay
from lib import MultiBoxEncoder
from lib import multibox_loss
from lib import preproc_for_train
from lib import SSD300
from lib import VOCDataset


class TrainWrapper(chainer.Chain):

    def __init__(self, model, k=3):
        super().__init__(model=model)
        self.k = k

    def __call__(self, x, t_loc, t_conf):
        loc, conf = self.model(x)
        loss_loc, loss_conf = multibox_loss(loc, conf, t_loc, t_conf, self.k)
        loss = loss_loc + loss_conf
        chainer.report(
            {'loss': loss, 'loc': loss_loc, 'conf': loss_conf}, self)
        return loss


class TrainDataset(chainer.dataset.DatasetMixin):

    def __init__(self, datasets, model):
        self.datasets = datasets
        self.insize = model.insize
        self.mean = model.mean
        self.encoder = MultiBoxEncoder(model)

    def __len__(self):
        return sum(map(len, self.datasets))

    def get_example(self, i):
        for dataset in self.datasets:
            if i >= len(dataset):
                i -= len(dataset)
                continue

            image = dataset.image(i)
            boxes, labels = dataset.annotations(i)
            image, boxes, labels = preproc_for_train(
                image, boxes, labels, self.insize, self.mean)
            loc, conf = self.encoder.encode(boxes, labels)
            return image, loc, conf


# To skip unsaved parameters, use strict option.
# This function will be removed.
def load_npz(filename, obj):
    with np.load(filename) as f:
        d = serializers.NpzDeserializer(f, strict=False)
        d.load(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='VOCdevkit')
    parser.add_argument('--train', action='append')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--output', default='result')
    parser.add_argument('--init')
    parser.add_argument('--resume')
    args = parser.parse_args()

    model = SSD300(20)
    if args.init:
        # To skip unsaved parameters, initialize model by default initializers.
        # This line will be removed.
        model(np.empty((1, 3, model.insize, model.insize), dtype=np.float32))
        load_npz(args.init, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    dataset = TrainDataset(
        [VOCDataset(args.root, *t.split('-')) for t in args.train], model)

    iterator = chainer.iterators.MultiprocessIterator(
        dataset, args.batchsize, n_processes=2)

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(TrainWrapper(model))
    optimizer.add_hook(CustomWeightDecay(0.0005, b={'lr': 2, 'decay': 0}))

    updater = training.StandardUpdater(iterator, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.output)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=0.001),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    snapshot_interval = 1000, 'iteration'
    log_interval = 10, 'iteration'

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/loc', 'main/conf', 'lr']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
