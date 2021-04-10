# WSSGG

* [1 Installation](#1-installation)
    - [1.1 Faster-RCNN](#11-faster-rcnn)
    - [1.2 Language Parser](#12-language-parser)
* [2 Settings)(#2-settings)
    - [2.1 VG-Gt-Graph](#21-vg-gt-graph)
    - [2.2 VG-Cap-Graph](#22-vg-cap-graph)
    - [2.3 COCO-Cap-Graph](#23-coco-cap-graph)
* [3 Training and Evaluation](#training)

## 1 Installation

```
git clone "https://github.com/yekeren/WSSGG.git" && cd "WSSGG"
```

We use Tensorflow 1.5 and Python 3.6.4. To continue, please ensure that at least the correct Python version is installed.
[requirements.txt](requirements.txt) defines the list of python packages we installed.
Simply run ```pip install -r requirements.txt``` to install these packages after setting up python.
Next, run ```protoc protos/*.proto --python_out=.``` to compile the required protobuf protocol files, which are used for storing configurations.

```
pip install -r requirements.txt
protoc protos/*.proto --python_out=.
```

### 1.1 Faster-RCNN
Our Faster-RCNN implementation relies on the [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
Users can use ```git clone "https://github.com/tensorflow/models.git" "tensorflow_models" && ln -s "tensorflow_models/research/object_detection" ``` to set up.
The specific model we use is [faster_rcnn_inception_resnet_v2_atrous_lowproposals_oidv2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz) to keep it the same as the [VSPNet](https://github.com/alirezazareian/vspnet). More information is in [Tensorflow object detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).

```
git clone "https://github.com/tensorflow/models.git" "tensorflow_models" 
ln -s "tensorflow_models/research/object_detection"
mkdir -p "zoo"
wget -P "zoo" "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz"
tar xzvf zoo/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz -C "zoo"
```

### 1.2 Language Parser
Though we indicate the dependency on spacy in [requirements.txt](requirements.txt), we still need to run ```python -m spacy download en''' for English.
Then, we checkout the tool at [SceneGraphParser](https://github.com/vacancy/SceneGraphParser) by running ```git clone "https://github.com/vacancy/SceneGraphParser.git" && ln -s "SceneGraphParser/sng_parser"'''

```
python -m spacy download en
git clone "https://github.com/vacancy/SceneGraphParser.git"
ln -s "SceneGraphParser/sng_parser"
```

## 2 Settings

### 2.1 VG-Gt-Graph

### 2.2 VG-Cap-Graph

### 2.3 COCO-Cap-Graph

## 3 Training and Evaluation


