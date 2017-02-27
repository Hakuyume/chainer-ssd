#! /usr/bin/env python3

import argparse

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions

import config
from ssd import SSD300
from multibox import MultiBoxEncoder
from loader import SSDLoader
from voc import VOC


class CustomHook(object):

    name = 'CustomHook'

    def __init__(self, decay):
        self.decay = decay

    def kernel(self):
        return chainer.cuda.elementwise(
            'T p, T lr, T decay', 'T g',
            'g = lr * g + decay * p',
            'custom_hook')

    def __call__(self, opt):
        for param in opt.target.params():
            if param.name == 'b':
                lr = 2
                decay = 0
            else:
                lr = 1
                decay = self.decay
            p, g = param.data, param.grad
            with chainer.cuda.get_device(p) as dev:
                if int(dev) == -1:
                    g = lr * g + decay * p
                else:
                    self.kernel()(p, lr, decay, g)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='VOCdevkit')
    parser.add_argument('--train', action='append')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--init')
    parser.add_argument('--resume')
    args = parser.parse_args()

    model = SSD300(n_classes=20, aspect_ratios=config.aspect_ratios)
    if args.init:
        serializers.load_npz(args.init, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    multibox_encoder = MultiBoxEncoder(
        grids=model.grids,
        steps=config.steps,
        sizes=config.sizes,
        aspect_ratios=model.aspect_ratios,
        variance=config.variance)

    train = SSDLoader(
        VOC(args.root, [t.split('-') for t in args.train]),
        model.insize,
        config.mean,
        multibox_encoder)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    optimizer = chainer.optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(CustomHook(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)

    snapshot_interval = 1000, 'iteration'
    log_interval = 10, 'iteration'

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'main/loc', 'main/conf', 'lr']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
