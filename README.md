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

## Training

## Evaluation
