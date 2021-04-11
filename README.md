# WSSGG

* [0 Overview](#0-Overview)
* [1 Installation](#1-installation)
    - [1.1 Faster-RCNN](#11-faster-rcnn)
    - [1.2 Language Parser](#12-language-parser)
* [2 Settings](#2-settings)
    - [2.1 VG-Gt-Graph and VG-Cap-Graph](#21-vg-gt-graph-and-vg-cap-graph)
    - [2.2 COCO-Cap-Graph](#22-coco-cap-graph)
* [3 Training and Evaluation](#training)
* [4 Visualization](#visualization)

## 0 Overview
Our model uses the image's paired caption as weak supervision to learn the entities in the image and the relations among them.
At inference time, it generates scene graphs without help from texts.
To learn our model, we first allow context information to propagate on the text graph to enrich the entity word embeddings (Sec. 3.1). 
We found this enrichment provides better localization of the visual objects.
Then, we optimize a text-query-guided attention model (Sec. 3.2) to provide the image-level entity prediction and associate the text entities with visual regions best describing them.
We use the joint probability to choose boxes associated with both subject and object (Sec. 3.3), then use the top scoring boxes to learn better grounding (Sec. 3.4).
Finally, we use an RNN (Sec. 3.5) to capture the vision-language common-sense and refine our predictions.

<img src="g3doc/images/overview.png">

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
Also, don't forget to using ```protoc``` to compire the protos used by the detection API.

The specific Faster-RCNN model we use is [faster_rcnn_inception_resnet_v2_atrous_lowproposals_oidv2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz) to keep it the same as the [VSPNet](https://github.com/alirezazareian/vspnet). More information is in [Tensorflow object detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).

```
git clone "https://github.com/tensorflow/models.git" "tensorflow_models" 
ln -s "tensorflow_models/research/object_detection"
cd tensorflow_models/research/; protoc object_detection/protos/*.proto --python_out=.; cd -

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

### 2.1 VG-Gt-Graph and VG-Cap-Graph

Typing ```sh dataset-tools/create_vg_settings.sh "vg-gt-cap"``` will generate VG-related files under the folder "vg-gt-cap" (for both VG-Gt-Graph and VG-Cap-Graph settings). Basically, it will download datasets and launch the following programs under the [dataset-tools](dataset-tools) directory:

| Name                                                                       | Desc.                                         |
|----------------------------------------------------------------------------|-----------------------------------------------|
| [create_vg_frcnn_proposals.py](dataset-tools/create_vg_frcnn_proposals.py) | Extract VG visual proposals using Faster-RCNN |
| [create_vg_text_graphs.py](dataset-tools/create_vg_text_graphs.py)         | Extract VG text graphs using Text Parser      |
| [create_vg_vocabulary](dataset-tools/create_vg_vocabulary.py)              | Get the VG vocabulary                         |
|                                                                            |                                               |


### 2.2 COCO-Cap-Graph

## 3 Training and Evaluation

## 4 Visualization
