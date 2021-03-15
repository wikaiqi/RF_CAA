# Conditional Adversarial Architecture 
### Tensorflow 2.0 version by Weikai Qi

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


 
