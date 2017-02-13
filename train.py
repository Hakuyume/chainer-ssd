#! /usr/bin/env python3

import argparse

import chainer
from chainer import training
from chainer.training import extensions

from ssd import SSD300
from multibox import MultiBoxEncoder
from voc import VOCDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--loaderjob', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    size = 300
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))

    multibox_encoder = MultiBoxEncoder(
        n_scale=6,
        variance=(0.1, 0.2),
        grids=(38, 19, 10, 5, 3, 1),
        aspect_ratios=aspect_ratios)

    model = SSD300(
        n_class=20,
        n_anchors=multibox_encoder.n_anchors)
    model.train = True
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    train = VOCDataset(args.root, size, multibox_encoder)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)

    optimizer = chainer.optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (1000, 'iteration'), args.out)

    snapshot_interval = 100, 'iteration'
    log_interval = 10, 'iteration'

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'lr']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
