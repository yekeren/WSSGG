# WSSGG

* [1. Installation](#1-installation)
    - [1.1 Faster-RCNN](#11-faster-rcnn)
* [Preparing datasets](#preparing-datasets)
    - [Set up the text parser from Stanford Scene Graph Parser](#setup-the-text-parser-from-stanford-scene-graph-parser)
    - [Set up experimental data following VSPNet](#set-up-experimental-data-following-vspnet)
    - [Set up experimental data of COCO captions](#set-up-experimental-data-of-coco-captions)
* [Training](#training)
* [Evaluation](#evaluation)

## 1. Installation

We use Tensorflow 1.5 and Python 3.6.4. To continue, please ensure that at least the correct Python version is installed.
[requirements.txt](requirements.txt) defines the list of python packages we installed.
Simply run ```pip install -r requirements.txt``` to install these packages after setting up python.
Next, run ```protoc protos/*.proto --python_out=.``` to compile the required protobuf protocol files, which are used for storing configurations.

### 1.1 Faster-RCNN
Our Faster-RCNN implementation relies on the [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
Users can use ```git clone "https://github.com/tensorflow/models.git" "tensorflow_models" && ln -s "tensorflow_models/research/object_detection" ``` to set up.
The specific model we use is [faster_rcnn_inception_resnet_v2_atrous_lowproposals_oidv2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz) to keep it the same as the VSPNet. More available models are listed in [Tensorflow object detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).

## Preparing datasets

### Setup the text parser from [Stanford Scene Graph Parser](https://nlp.stanford.edu/software/scenegraph-parser.shtml)
We use schuster et al. 2015 to generate text scene graphs from captions. Thus, the first step is to set up the provided text parser. The following command shall download the text parser as well as dependancies from [Stanford Scene Graph Parser](https://nlp.stanford.edu/software/scenegraph-parser.shtml). It will create a "tools/stanford-corenlp-full-2015-12-09" folder and put any required files in it.
```
sh tools/download_scene_graph_parser.sh tools

cd "tools/"
javac -cp "stanford-corenlp-full-2015-12-09/*:." "SceneGraphDemo.java"
java -mx2g -cp "stanford-corenlp-full-2015-12-09/*:." "SceneGraphDemo"
```
The above java commands will launch the examplar program, if you enter "both of the men are riding their horses" in the prompt java command line, the program will return:
```
(python) -bash-4.2$ java -mx2g -cp "stanford-corenlp-full-2015-12-09/*:." "SceneGraphDemo"
Processing from stdin. Enter one sentence per line.
> both of the men are riding their horses
{"relationships":[{"predicate":"have","subject":0,"text":["man","have","horse"],"object":2},{"predicate":"ride","subject":0,"text":["man","ride","horse"],"object":2},{"predicate":"have","subject":1,"text":["man","have","horse"],"object":3},{"predicate":"ride","subject":1,"text":["man","ride","horse"],"object":3}],"phrase":"both of the men are riding their horses","objects":[{"names":["man"]},{"names":["man"]},{"names":["horse"]},{"names":["horse"]}],"attributes":[],"id":0,"url":""}
```

### Extract text graphs from the VG captions
We use the following command to extract text graphs from the VG region descriptions. It'll download the region descriptions from the VG dataset and run schuster's parser.
```
sh tools/download_and_preprocess_vg_captions.sh data-vspnet/text_graphs
```

###  Set up experimental data following [VSPNet](https://github.com/alirezazareian/vspnet)
We provide scripts and tools to set up experiments identical to the Zareian et al., CVPR2020. We download proposal boxes, box features, and preprocessed data annotation splits from their git repository. For more information, please refer to [their repository](https://github.com/alirezazareian/vspnet).

The following scripts shall download the data needed and generate .tfrecord files under the "./data-vspnet/tfrecords" directory.

```
sh download_and_prepare_vspnet_experiments.sh "data-vspnet"
```

### Set up experimental data of COCO captions

```
sh "tools/download_and_preprocess_mscoco.sh" "data-mcoco"
```

## Training

## Evaluation

it is changed.
