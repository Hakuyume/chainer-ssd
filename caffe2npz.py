#! /usr/bin/env python3

import argparse

from chainer import serializers

from lib import load_caffe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel')
    parser.add_argument('output')
    args = parser.parse_args()

    model = load_caffe(args.caffemodel)
    serializers.save_npz(args.output, model)
