#! /usr/bin/env python3

import argparse
import numpy as np

from chainer import serializers

from lib import load_caffe
from lib import SSD300


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel')
    parser.add_argument('output')
    parser.add_argument('--n_classes', type=int, default=20)
    parser.add_argument('--base_only', action='store_true')
    parser.set_defaults(base_only=False)
    args = parser.parse_args()

    model = SSD300(args.n_classes)
    load_caffe(args.caffemodel, model, base_only=args.base_only)
    if args.base_only:
        x = np.empty((1, 3, model.insize, model.insize), dtype=np.float32)
        model(x)

    serializers.save_npz(args.output, model)
