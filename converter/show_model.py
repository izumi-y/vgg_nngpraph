#coding:utf-8

import chainer
import os
import sys
import argparse
import numpy as np
import cv2 as cv
import cPickle as pickle
from PIL import Image
from VGG16 import VGG16
from chainer import cuda
from chainer import serializers
from chainer import Variable
from chainer import optimizers

from show import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
xp = cuda.cupy
vgg = VGG16()
serializers.load_hdf5('VGG.model', vgg)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    vgg.to_gpu()

#show_model(vgg)

imsize = 224
batchsize = 1
x = xp.zeros((batchsize, 3, imsize, imsize), dtype=xp.float32)
y = xp.zeros((batchsize,), dtype=xp.int32)
x[0][0] -= 103.939
x[0][1] -= 116.779
x[0][2] -= 123.68

x = Variable(x, 0, "x")
y = Variable(y, 0, "y")

loss = vgg(x, y)


#print(loss.creator.label)
#print(loss.creator.inputs[0].creator.inputs[0].shape)#last fc layer


#a = loss.creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[1].shape
#b = loss.creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].shape
#print(a)
#print(b)


#print(loss.creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[1].data)#first convlayer


#for x in dir(a):
#    print x


'''
#print(loss.creator.inputs[0].creator.label)  # layer_name
#print(loss.creator.inputs[0].creator.inputs[0])   #input
#print(loss.creator.inputs[0].creator.inputs[1])   #Weight
#print(loss.creator.inputs[0].creator.inputs[2])   #bias

# y = inputs[0] * inputs[1] + inputs[2]
'''

#print (loss.creator.inputs[0].creator.label)


#old
#b = []
#show_func(loss, b)

#new
make_func_list_ver2(loss)
















