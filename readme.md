# Conditional Adversarial Architecture 
### Tensorflow 2.0 version by Weikai Qi


## Part 1: Binary classification using a CNN model

![Figure 1](./pictures/spectrogram.png)

In the code, every raw sensor recording is converted to a two-dimensional spectrogram. To avoid regenerate the 2d representations of original data every time, I introduce a parameter `recalculate`.  By setting the parameter `recalculate` to false, the code will generate 2d representation files only if representation files don't exist. I split the data into train / validation / test datasets with a fraction of 80%, 10%, 10% respectively. The following is a table that lists the number of samples with label=0 (F) and label = 1 (MW), and the number of samples from subjects 1 and 2: 
| datasets | samples with label=0 (F) | samples with label=1 (MW) | subject 1 samples | subject 2 samples |
|-----------|:-------------------------:|:--------------------------:|:-------------------:|:------------------:|
| training | 377 | 245 | 420 | 302 |
| validation | 44 | 46 | 51 | 39|
|test | 47 | 44 | 57 | 34 |


The ratio between two classes (label=0 or 1) are 1.5, 0.96, 1.02 for training, validation, and test, and The ratio between two subjects are 1.39, 1.3, 1.67 for the training, validation, and test. Subject 1 has more data than subject 2, which means an algorithm has a chance bias to subject 1, for example, if we use a vanilla CNN model.  

### Run the CNN model

 Similar to [Zhao etc.'s paper](http://sleep.csail.mit.edu), I adopt an encoder network which is composed of 16 convolutional layers with residual blocks. The size of the input data is (224,224, 1). Since there are only several hundred samples, I reduce the number of layers and filters to avoid overfitting. The predictor has two fully connected (FC) networks. The encoder plus the predictor is a CNN model has similar architecture as ResNet.  

Before start running the code, the first step is to copy `file_locator.csv` and `intermediate` folder to the `data` folder in this repo.

Train the CNN model: 
```
python train.py -p data/ --train true
```
By default, it will run 120 epochs. We can change it to other value (for example, 200) by adding `--epochs 200`. 

Load pre-train model and run prediction: 
```
python train.py --checkpoint "check_point-CNN"
```

### Performance of the CNN model
The performance of the CNN model (averaged by 5 runs):
| datasets | Accuracy (sd) | 
|-----------|:-------------------------:|
| training | 98.2% (3.6%) | 
| validation | 79.8% (1.6%) |
|test | 79.7% (3.2%) | 

 The accuracy of the training set is much higher than the validation set and the test set, which indicates the model is overfitting. Including more data through data augmentation or reducing the size of the model can potentially improve performance. However, overfitting is also potentially relating to the noise in data. Since the raw data is collected from two subjects in different environments. The background noise might mistakenly be recognized as patterns by a CNN model. Remove those subject-dependent noises in the data will help us to build a better subject-independent solution. 


## Part 2: Three-Way Game

Zhao's paper points out that if we use a discriminator, in theory, it will discard extraneous information about the source (in this case it is the subject-dependent noise) when the three-players game is at the equilibrium. Each subject's background noise will be removed from the signal by adding the adversarial training process, which will maximize the discriminator loss and minimize the predictor loss. The predictor guides the encoder to learn how to do the classification, and the discriminator guides the encoder to learn a subjects-independent feature map. Since the output of the predictor plays as an underlying posterior, we should avoid the gradient backpropagation from the discriminator to the predictor. More details about this three-way game can be found in the paper. In this code, the discriminator is two layers FC networks with the same architecture as the predictor. 

Train the three-way game model: 
```
python train.py --train true --game true
```

Load pre-train model and run prediction: 
```
python train.py --checkpoint "check_point-CAA"
```

The performance of the three-way game model (averaged by 5 runs): 
| datasets | Accuracy (sd) | 
|-----------|:-------------------------:|
| training | 88.4% (2.5%) | 
| validation | 83.3% (1.8%) |
|test | 85.5% (1.4%) | 


The performance of the three-way game model on the test dataset is better than the CNN model. The discriminator of the three-way game model helps to remove subject-dependent information in feature maps. To prove this scenario, I generate the TSNE plot of the feature map using the last layer of the encoder. 

![Figure 2](./pictures/tsne_CAA.png) ![Figure 3](./pictures/tsne_CNN.png)

The three-way game model generates a better feature map. We can see the majority of samples of class=0 or 1 forms a cluster and no matter the data from subject 1 or 2 in the TSNE plot of the three-way game model. But it's worth to mention that the majority data of the CNN model also form clusters in the TSNE plot, but it could be due to the limited amount of data. All the conclusion make in this study, need to confirm with a larger size of data. 


## Requirement

The list of required packages can be found in `requirements/requirements.txt`. There are different ways to get the required package:
 - Run `pip install ` to get packages in the list of `requirements/requirements.txt`. 
 - if you use `conda`, you can use the `environment.yaml`:
 ``` 
 conda env create -f environment.yaml
 ```
 It will create a new environment named `CAA`, and then run `coda activate CAA` to activate the new environment.

## Run the code on Colab
You can copy the code and data to google drive and run it on colab. Here is an example code you can add in a colab notebook and it will train a cnn model:
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
%cd /content/gdrive/My Drive/CAA/RFSleep_CAA/  ## change this path to where you put the data
!python train.py --train true --epochs 120
```
We can access to a GPU/TPU card with a Colab PRO account. For example, collab assigns a 'Tesla P100-PCIE-16GB' if I choice 'GPU' in the notebook setting, and it takes less than 10 min to train the model.  

##  Parameters
```
usage: train.py [-h] [-p PATH] [-n EPOCHS] [--l2 L2] [--game GAME] [-l GAME_LAMBDA] [--train TRAIN]
                [--recalculate RECALCULATE] [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  data file path
  -n EPOCHS, --epochs EPOCHS
                        number of epochs
  --l2 L2               l2 parameter
  --game GAME           if game_on = ture, run the three-way game; otherwise, run a CNN model
  -l GAME_LAMBDA, --game_lambda GAME_LAMBDA
                        the 3-way game parameter lambda
  --train TRAIN         if true, train model; otherwise, run prediction use checkpoint
  --recalculate RECALCULATE
                        if true, regenerate spectrogram data
  --checkpoint CHECKPOINT
                        the path of checkpoint
```     

## References
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition, CVPR,  2016. 

[2] Mingmin Zhao, Shichao Yue, Dina Katabi, Tommi Jaakkola, Matt Bianchi, Learning Sleep Stages from Radio Signals: A Conditional Adversarial Architecture, ICML, 2017.


 
