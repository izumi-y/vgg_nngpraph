# -*- coding: utf-8 -*-

import os
import math
import sys
import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
import numpy as np
from show import *

b = []


conv_no = 1
pool_no = 1
fc_no = 1
tmp = ""
tmp2 = ""
n = 0
layer_type_no = -1

#Parameter保存場所
dest_path = "Parameter/param1225"


#モデル表示
def show_model(src):
#    assert isinstance(src, chainer.Chain)
    for child in src.children():
        if isinstance(child, chainer.Chain):
            print "[Chain] ",child.name
            show_model(child)
        if isinstance(child, chainer.Function):
            print "[Func] ",child.name
        if isinstance(child, chainer.Link):
            print "[Link] ",child.name
            for a in (child.namedparams()):
                print "shape: ",a[1].data.shape




#ここからしばらく過去の記録

def show_func(v, b):
    a = v.creator
    l = a.label

    if a.rank > 0:
        show_func(a.inputs[0], b)
    
    b.append(l)
    #print l

def show_func_ver1(v):
    global b
    a = v.creator
    l = a.label

    if a.rank > 0:
        show_func_ver1(a.inputs[0])
    
    b.append(l)
    print (b)




def make_func_list(src):
    global conv_no, pool_no, fc_no, tmp, tmp2
    a = src.creator
    l = a.label

    if a.rank > 0:
        make_func_list(a.inputs[0])
    

    if l == "Convolution2DFunction":
        tmp = "conv" + str(conv_no)
        tmp2 = "conv" + str(conv_no - 1)
        b.append([tmp,a.inputs[1].shape])

        if a.rank == 0:
            print('let %s = MPSCNNConvolutionNode(sorce: scale.resultImage, weights: DataSource("%s", %d, %d, %d, %d))' % (tmp, tmp, a.inputs[1].shape[2], a.inputs[1].shape[3], a.inputs[1].shape[1], a.inputs[1].shape[0]))
            print(b[0][1])
        else:
            print('let %s = MPSCNNConvolutionNode(sorce: %s.resultImage, weights: DataSource("%s", %d, %d, %d, %d))' % (tmp, tmp2, tmp, a.inputs[1].shape[2], a.inputs[1].shape[3], a.inputs[1].shape[1], a.inputs[1].shape[0]))
            print(" ")

        conv_no += 1
        

    if l == "ReLU":
        tmp = "ReLU"
        

    if l == "MaxPooling2D":
        tmp = "conv" + str(conv_no)
        #tmp2 = "conv" + str(conv_no - 1)
        b.append([tmp, a.rank])
        print('let %s = MPSCNNPoolingMaxNode(sorce: %s.resultImage, filterSize: 2)' % (tmp, tmp2)) #can not find filterSize 
        print(" ")
        pool_no += 1
       

    if l == "LinearFunction":
        tmp = "fc" + str(fc_no)
        tmp2 = "fc" + str(conv_no - 1)
        b.append([tmp, a.rank])
        
        if a.rank == 0:
            print('let %s = MPSCNNFullyConnectedNode(sorce: scale.resultImage, weights: DataSource("%s", %d, %d, %d, %d))' % (tmp, tmp, a.inputs[1].shape[2], a.inputs[1].shape[3], a.inputs[1].shape[1], a.inputs[1].shape[0]))
            print(" ")
        else:
            print('let %s = MPSCNNFullyConnectedNode(sorce: %s.resultImage, weights: DataSource("%s", 1, 1, %d, %d))' % (tmp, tmp2, tmp, a.inputs[1].shape[1], a.inputs[1].shape[0]))
            print(" ")

        fc_no += 1
        

    if l == "Softmax":
        tmp = "Softmax"
        
    
    #print (b)







"""
def show_func2(v,b,net):
    a = v.creator
    l = a.label
    
    if l == "Convolution2D":
        print(let conv%d = MPSCNNConvolutionNode(sorce: %s, weights: DataSource("conv%d", %d, %d, %d, %d)), cv, 


"""
#過去の記録ここまで（12/8）





#レイヤーのリストを作成するやつ12/8
"""
Layer_type_no
1 ---> Convolution 
2 ---> MaxPooling
3 ---> FullyConnection(LinearFunction)
"""
"""
ｂの中身
conv ---> [0]conv1,conv2...  [1]layer_type_no(1)  [2](128,64,3,3)...  [3]weights   [4]bias
pool ---> [0]pool1,...       [1]layer_type_no(2)  [2]レイヤー番号(rank)
fc   ---> [0]fc1,...         [1]layer_type_no(3)  [2](1000,4096)...   [3]weights   [4]bias
"""



def make_func_list_ver2(src):
    global b, conv_no, pool_no, fc_no, tmp, layer_type_no, n
    a = src.creator
    l = a.label

    if a.rank > 0:
        make_func_list_ver2(a.inputs[0])
    

    if l == "Convolution2DFunction":
        tmp = "conv" + str(conv_no)
        layer_type_no = 1 
        b.append([tmp, layer_type_no, a.inputs[1].shape, a.inputs[1].data, a.inputs[2].data])

        conv_no += 1
        

    if l == "ReLU":
        tmp = "ReLU"
        

    if l == "MaxPooling2D":
        tmp = "pool" + str(pool_no)
        layer_type_no = 2
        b.append([tmp, layer_type_no, a.rank])
        
        pool_no += 1
       

    if l == "LinearFunction":
        tmp = "fc" + str(fc_no)
        layer_type_no = 3
        b.append([tmp, layer_type_no, a.inputs[1].shape, a.inputs[1].data, a.inputs[2].data])

        fc_no += 1
        

    if l == "Softmax":
        tmp = "Softmax"
        n = 1

#最終レイヤーの判定が一般化出来てないから暫定で
    if n == 1:
        #show_swift_func(b)
        make_param_file(b)  

    




#レイヤーのリストからMPSCNNのprintするやつ12/8

def show_swift_func(func_list):
    #print(func_list)


    for (i,x) in enumerate(func_list):

        if x[1] == 1:
            if i == 0:
                print('let %s = MPSCNNConvolutionNode(sorce: scale.resultImage, weights: DataSource("%s", %d, %d, %d, %d))' 
                      % (x[0], x[0], x[2][2], x[2][3], x[2][1], x[2][0]))
            else:
                print('let %s = MPSCNNConvolutionNode(sorce: %s.resultImage, weights: DataSource("%s", %d, %d, %d, %d))' 
                      % (x[0], func_list[i-1][0], x[0], x[2][2], x[2][3], x[2][1], x[2][0]))


        if x[1] == 2:
            if i == 0:
                print('let %s = MPSCNNPoolingMaxNode(sorce: scale.resultImage, filterSize: 2)' % (x[0])) #filter size 2 only
            else:
                print('let %s = MPSCNNPoolingMaxNode(sorce: %s.resultImage, filterSize: 2)' % (x[0], func_list[i-1][0]))


        if x[1] == 3:
            if i == 0:
                print('let %s = MPSCNNFullyConnectedNode(sorce: scale.resultImage, weights: DataSource("%s", 1, 1, %d, %d))' 
                       % (x[0], x[0], x[2][1], x[2][0]))
            else:
                print('let %s = MPSCNNFullyConnectedNode(sorce: %s.resultImage, weights: DataSource("%s", 1, 1, %d, %d))' 
                       % (x[0], func_list[i-1][0], x[0], x[2][1], x[2][0]))



    print("--------------------------------")
    print("--------------------------------")
    print(i,x)




#レイヤーのリストからパラメータファイル(.bin)を出力する12/8
#1225fix


def make_param_file(func_list):
    for (i, x) in enumerate(func_list):
        if x[1] == 1:      #conv
            conv_weights = x[3]
            conv_bias = x[4]
            #print(conv_weights.shape)
            #print(conv_bias.shape)
            conv_weights = conv_weights.transpose(0, 2, 3, 1)
            #print(conv_weights.shape)
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
            conv_weights.tofile(os.path.join(dest_path, x[0] + "_w" + ".bin"))
            conv_bias.tofile(os.path.join(dest_path, x[0] + "_b" + ".bin"))
        if x[1] == 3:       #fc
            fc_weights = x[3]
            fc_bias = x[4]
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
            fc_weights.tofile(os.path.join(dest_path, x[0] + "_w" + ".bin"))
            fc_bias.tofile(os.path.join(dest_path, x[0] + "_b" + ".bin"))



