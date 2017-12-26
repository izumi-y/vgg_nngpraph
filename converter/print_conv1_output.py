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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--image', type=str, default='dog.jpg')
args = parser.parse_args()
xp = cuda.cupy
vgg = VGG16()
serializers.load_hdf5('VGG.model', vgg)

vgg.conv1_1.b.data = np.zeros_like(vgg.conv1_1.b.data)

dest_path = "memo"

mean = np.array([103.939, 116.779, 123.68])

img = cv.imread(args.image).astype(np.float32)
img -= mean
img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
img = img[np.newaxis, :, :, :]


cuda.get_device(args.gpu).use()
vgg.to_gpu()
img = cuda.cupy.asarray(img, dtype=np.float32)



loss = vgg(Variable(img), None)


hoge = loss.creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].creator.inputs[0].data


'''
#print(chainer.cuda.to_cpu(hoge)[0].shape)
f = open('conv1_out_1225.txt','w')
for i,j in enumerate(chainer.cuda.to_cpu(hoge)[0]):
    f.write(str(j) + "\n")
	#print(i, j)

f.close()


f = open('conv1_out_ver2.txt','w')
for i,j in enumerate(chainer.cuda.to_cpu(hoge)[0]):
    for m,n in enumerate(j):
        f.write(str(n) + "\n")

f.close()

f = open('conv1_out_1225_ver3.txt','w')
for i in range(len(hogehoge)):
    if i%10 == 0:
        f.write(str(hogehoge[i]) + "\n")
    else:
        f.write(str(hogehoge[i]))
f.close()
'''

print (hoge[0][0][0])

#hogehoge = hoge.flatten()

#for x in dir(vgg):
 #   print x


#print(vgg.h1.data) 

