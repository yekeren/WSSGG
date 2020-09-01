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

We provide scripts and tools to set up experiments identical to the Zareian et al., CVPR2020. We download proposal boxes, box features, and preprocessed data annotation splits from their git repository. For more information, please refer to [their repository](https://github.com/alirezazareian/vspnet).

The following scripts shall download the data needed and generate .tfrecord files under the "./data-vspnet/tfrecords" directory.

```
sh download_and_prepare_vspnet_experiments.sh "data-vspnet"
```

To check the validity of the generated .tfrecord files, run 

```
python "readers/scene_graph_reader_demo.py" \
  --image_directory "data-vspnet/images"
  --tf_record_file PATH_TO_THE_FILE
```

## Training

## Evaluation
