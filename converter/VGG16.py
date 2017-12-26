import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


class VGG16(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = False

    def __call__(self, x, t):
        h1 = F.relu(self.conv1_1(x))
        h2 = F.relu(self.conv1_2(h1))
        h3 = F.max_pooling_2d(h2, 2, stride=2)

        h4 = F.relu(self.conv2_1(h3))
        h5 = F.relu(self.conv2_2(h4))
        h6 = F.max_pooling_2d(h5, 2, stride=2)

        h7 = F.relu(self.conv3_1(h6))
        h8 = F.relu(self.conv3_2(h7))
        h9 = F.relu(self.conv3_3(h8))
        h10 = F.max_pooling_2d(h9, 2, stride=2)

        h11 = F.relu(self.conv4_1(h10))
        h12 = F.relu(self.conv4_2(h11))
        h13 = F.relu(self.conv4_3(h12))
        h14 = F.max_pooling_2d(h13, 2, stride=2)

        h15 = F.relu(self.conv5_1(h14))
        h16 = F.relu(self.conv5_2(h15))
        h17 = F.relu(self.conv5_3(h16))
        h18 = F.max_pooling_2d(h17, 2, stride=2)

        h19 = F.dropout(F.relu(self.fc6(h18)), train=self.train, ratio=0.5)
        h20 = F.dropout(F.relu(self.fc7(h19)), train=self.train, ratio=0.5)
        h21 = self.fc8(h20)
        
        if self.train:
            self.loss = F.softmax_cross_entropy(h21, t)
            self.acc = F.accuracy(h21, t)
            return self.loss
        else:
            self.pred = F.softmax(h21)
            return self.pred
        

        
