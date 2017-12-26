# vgg_nngpraph

converter/data/VGG_ILSVRC_16_layers.caffemodel

converter/data/VGG_ILSVRC_16_layers_deploy.prototxt

を準備


```
$ python load_model.py
```
でVGG16.modelを作成


```
$ python make_conv0.py
```
でRGB2GBRの変換等を行うレイヤーのweightとbiasファイルの生成


```
$ python predict.py　--image dog.jpg
```
で1000種類画像認識


```
$ python print_conv1_out.py
```
で最初のconvolution層の出力を出力


```
$ python show_model.py
```
でvggの各レイヤーのweightとbiasファイルを生成

dest_pathで出力先を指定

