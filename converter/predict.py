#coding:utf-8

import chainer
import argparse
import numpy as np
import cv2 as cv
from PIL import Image
import cPickle as pickle
from VGG16 import VGG16
from chainer import cuda
from chainer import serializers
from chainer import Variable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--image', type=str, default='dog.jpg')
    args = parser.parse_args()

#use PIL
    mean = np.array([123.68, 116.779, 103.939])#pil
    img = Image.open(args.image)
    img = img.resize((224,224))
    img -= mean
    #print(img.shape)
    img = img[np.newaxis, :, :, :]
    #print(type(img))
    img = img[:, :, :, ::-1]  #rgb->bgr
    img = img.transpose(0, 3, 1, 2)


#use openCV
    #mean = np.array([103.939, 116.779, 123.68])
    #img = cv.imread(args.image).astype(np.float32)
    #img -= mean
    #img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
    #img = img[np.newaxis, :, :, :]
    #img = img[:, :, :, ::-1]  #brg->rgb
    
    #print(img.shape)

    vgg = VGG16()
    serializers.load_hdf5('VGG.model', vgg)



#bias->0
    #print(vgg.conv1_1.b.data)
    #print(type(vgg.conv1_1.b.data))
    #vgg.conv1_1.b.data = np.zeros_like(vgg.conv1_1.b.data)
    #print(vgg.conv1_1.b.data)
    #vgg.conv1_2.b.data = np.zeros_like(vgg.conv1_2.b.data)
    #vgg.conv2_1.b.data = np.zeros_like(vgg.conv2_1.b.data)
    #vgg.conv2_2.b.data = np.zeros_like(vgg.conv2_2.b.data)
    #vgg.conv3_1.b.data = np.zeros_like(vgg.conv3_1.b.data)
    #vgg.conv3_2.b.data = np.zeros_like(vgg.conv3_2.b.data)
    #vgg.conv3_3.b.data = np.zeros_like(vgg.conv3_3.b.data)
    #vgg.conv4_1.b.data = np.zeros_like(vgg.conv4_1.b.data)
    #vgg.conv4_2.b.data = np.zeros_like(vgg.conv4_2.b.data)
    #vgg.conv4_3.b.data = np.zeros_like(vgg.conv4_3.b.data)
    #vgg.conv5_1.b.data = np.zeros_like(vgg.conv5_1.b.data)
    #vgg.conv5_2.b.data = np.zeros_like(vgg.conv5_2.b.data)
    #vgg.conv5_3.b.data = np.zeros_like(vgg.conv5_3.b.data)
    #vgg.fc6.b.data = np.zeros_like(vgg.fc6.b.data)
    #vgg.fc7.b.data = np.zeros_like(vgg.fc7.b.data)
    #vgg.fc8.b.data = np.zeros_like(vgg.fc8.b.data)





    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        vgg.to_gpu()
        img = cuda.cupy.asarray(img, dtype=np.float32)

    pred = vgg(Variable(img), None)

    if args.gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data


    words = open('synset_words.txt').readlines()
    words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
    words = np.asarray(words)


    #tmp1 = open('synset_words.txt').readlines()
    #tmp2 = [''.join(l[1:]) for l in [l.split() for l in tmp1]]
    #print(tmp2)

    top5 = np.argsort(pred)[0][::-1][:5]
    probs = np.sort(pred)[0][::-1][:5]
    for w, p in zip(words[top5], probs):
        print('{}\tprobability:{}'.format(w, p))

