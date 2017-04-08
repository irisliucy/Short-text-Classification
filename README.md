## Short Text Classification with Deep Neural Networks: An Exerimental Analysis

An experimental analysis with the aim to examine different neural networks and machine learning approaches. We also include a structural model that comprises of a temporal LSTM and a machine learning classifier. We extract the intermediate outputs from the hidden layers of a LSTM for feature engineering in a machine learning classifer. We run our models on 3 different datasets, [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/), [Ohsumed dataset](http://disi.unitn.it/moschitti/corpora.htm) and [Yahoo! Answer dataset](https://cogcomp.cs.illinois.edu/page/resource_view/89). In order to solve a multi-class classification problem, we created benchmark datasets based on those source datasets.

### Requirements
Code is written in Python (2.7) and requires Keras (2.0.2) with theano backend.

Using the pre-trained `Glove` word embedding vectors (pre-trained on Twitter / Wikipedia corpus) will also require downloading the binary file from
https://nlp.stanford.edu/projects/glove/ 

### Using the GPU
GPU is highly recommended to run the Keras models. We use two machines equipped with GeForce GTX 950 and took ~5-8 hours for a 20-class dataset. I believe you can speed things up with better GPUs.

To run on GPUs, you will need a `theano.rc` file in your home directory. Inside the file, you have to set `device = gpu` to enable the GPU.
```
[global]
device = gpu
floatX = float32
optimizer_including = cudnn
allow_gc = True

[nvcc]
flags=-D_FORCE_INLINES

[cuda]
root = /usr/local/cuda-8.0

[lib]
cnmem = 0.7
```


To choose Theano as Keras backend, you will need a Keras folder with a `keras.json` file. Insider the json file, you need to specify
```
{
    "image_dim_ordering": "th", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
```




