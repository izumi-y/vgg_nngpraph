#coding:utf-8
import numpy as np
import os


dest_path = "Parameter/conv0/param1226_3"

w = [[]]
#b = [123.68, 116.779, 103.939]  #rgb
b = [103.939, 116.779, 123.68]  #bgr

w.append([])
w.append([])

w[0].append([[0.0]])
w[0].append([[0.0]])
w[0].append([[255.0]])

w[1].append([[0.0]])
w[1].append([[255.0]])
w[1].append([[0.0]])

w[2].append([[255.0]])
w[2].append([[0.0]])
w[2].append([[0.0]])


#print(np.shape(w))

#print(np.array(b))

w2 = np.array(w, dtype = "float32")
print(w2.shape)
w2 = w2.transpose(0, 2, 3, 1)
print(w2.shape)
b2 = np.array(b, dtype = "float32")
print(b2.shape)

if not os.path.isdir(dest_path):
    os.mkdir(dest_path)
np.array(w2).tofile(os.path.join(dest_path, "conv0_w" + ".bin"))
np.array(b2).tofile(os.path.join(dest_path, "conv0_b" + ".bin"))



#1226_3 rgb->bgr b-bgr 255
#1226_4 rgb->bgr b-bgr 1
#1226_5 rgb->rgb b-rgb 255
