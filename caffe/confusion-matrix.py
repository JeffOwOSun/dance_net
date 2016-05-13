#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import caffe
import numpy as np
import argparse
from collections import defaultdict

TRAIN_DATA_ROOT=''

if __name__ == "__main__":
    '''parse the arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--meanfile', type=str, required=True)
    parser.add_argument('--labelfile', type=str, required=True)
    args = parser.parse_args()

    '''meanfile'''
    proto_data = open(args.meanfile, 'rb').read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]
    print dir(mean)
    print mean.shape

    '''the network, from path to prototxt'''
    net = caffe.Classifier(args.proto, args.model,
                       mean=mean,
                       #channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

    caffe.set_mode_gpu()

    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    #net = caffe.Net(args.proto, args.model, caffe.TEST)

    f = open(args.labelfile, "r")
    for line in f.readlines():
        #print 'line',line
        parts = line.split()
        example_image = parts[0]
        label = int(parts[1])
        #print 'label', label
        input_image = caffe.io.load_image(TRAIN_DATA_ROOT + example_image)
        prediction = net.predict([input_image])
        plabel = int(prediction[0].argmax())
        count += 1
        iscorrect = label == plabel
        correct += (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])
        #if not iscorrect:
        #    print("\rError: expected %i but predicted %i" \
        #            % (label, plabel))
        if not count % 100:
            print 'processed {}'.format(count)

        sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        sys.stdout.flush()

    print(", %i/%i corrects" % (correct, count))

    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])
