# WSSGG

* [Preparing code base](#preparing-code-base)
* [Preparing datasets](#preparing-datasets)
* [Training](#training)
* [Evaluation](#evaluation)

## Preparing code base

We use Tensorflow 1.5 and Python 3.6.4. A list of python package installed can be found in the [requirements.txt](requirements.txt). Simply, run the following command to install packages after setting up python.

```
pip install -r "requirements.txt"
```

Our Fast-RCNN implementation relies on the [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). So, run the following command to clone their repository.

```
sh prepare.sh
```

Then, we need to compile the .proto files. Our program uses .proto files to store configurations.

```
sh build.sh
```

## Preparing datasets

### Setup the text parser from [Stanford Scene Graph Parser](https://nlp.stanford.edu/software/scenegraph-parser.shtml).
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
source              reln                target              
---                 ----                ---                 
man-4               have                horse-8             
man-4               ride                horse-8             
man-4'              have                horse-8'            
man-4'              ride                horse-8'            


Nodes               
---                 
man-4               
man-4'              
horse-8             
horse-8'            

{"relationships":[{"predicate":"have","subject":0,"text":["man","have","horse"],"object":2},{"predicate":"ride","subject":0,"text":["man","ride","horse"],"object":2},{"predicate":"have","subject":1,"text":["man","have","horse"],"object":3},{"predicate":"ride","subject":1,"text":["man","ride","horse"],"object":3}],"phrase":"both of the men are riding their horses","objects":[{"names":["man"]},{"names":["man"]},{"names":["horse"]},{"names":["horse"]}],"attributes":[],"id":0,"url":""}
```

We provide scripts and tools to set up experiments identical to the Zareian et al., CVPR2020. We download proposal boxes, box features, and preprocessed data annotation splits from their git repository. For more information, please refer to [their repository](https://github.com/alirezazareian/vspnet).

The following scripts shall download the data needed and generate .tfrecord files under the "./data-vspnet/tfrecords" directory.

```
sh download_and_prepare_vspnet_experiments.sh "data-vspnet"
```

To check the validity of the generated .tfrecord files, run 

```
python "readers/scene_graph_reader_demo.py" \
  --image_directory "data-vspnet/images" \
  --tf_record_file PATH_TO_THE_FILE
```

## Training

## Evaluation

it is changed.
