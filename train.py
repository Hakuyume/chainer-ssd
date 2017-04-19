#! /usr/bin/env python3

import argparse

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from lib import MultiBoxEncoder
from lib import multibox_loss
from lib import preproc_for_train
from lib import SSD300
from lib import VOCDataset


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


class SSDTrainer(chainer.Chain):

    def __init__(self, model):
        super().__init__(model=model)

    def __call__(self, x, t_loc, t_conf):
        loc, conf = self.model(x)
        loss_loc, loss_conf = multibox_loss(loc, conf, t_loc, t_conf)
        loss = loss_loc + loss_conf
        chainer.report(
            {'loss': loss, 'loc': loss_loc, 'conf': loss_conf},  self)
        return loss


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

    model = SSD300(20)
    if args.init:
        serializers.load_npz(args.init, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    dataset = TrainDataset(
        [VOCDataset(args.root, *t.split('-')) for t in args.train], model)

    iterator = chainer.iterators.SerialIterator(dataset, args.batchsize)

    optimizer = chainer.optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(SSDTrainer(model))
    optimizer.add_hook(CustomHook(0.0005))

    updater = training.StandardUpdater(iterator, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1),
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
