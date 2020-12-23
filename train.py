#!/usr/bin/env python
#-----------------------------------------------
# Conditional Adversarial Architecture 
# Three way games (E, F, D)
# Simple ResNet version
# Author: Weikai Qi
# Email: wikaiqi@gmail.com
# Reference: M.Zhao etc.,ICML 2017
#-----------------------------------------------

import argparse
from src.tsne_plots import plot_tsne
from src.model import train_model, predict_model
from src.dataset import read_dataset


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default='data/', help="data file path")
    parser.add_argument("-n", "--epochs", type=int, default=120, help="number of epochs") 
    parser.add_argument("--l2", type=float, default=0.01, help="l2 parameter")
    parser.add_argument("--game", type=bool, default=False, help="if game_on = ture, run three way games; otherwise, run a CNN model")
    parser.add_argument("-l", "--game_lambda", type=float, default=1.3, help="3-way game parameter lambda")
    parser.add_argument("--train", type=bool, default=False, help="if true, train model; else run prediction use checkpoint")
    parser.add_argument("--recalculate", type=bool, default=False, help="if true, regenerate spectrogram data")
    parser.add_argument("--checkpoint", type=str, default='check_point-CNN', help="if true, regenerate spectrogram data")
    arg = parser.parse_args()

    print('----------------------------------------------------------')
    if arg.train:
        feature_map, label, game_loss = train_model(arg.path, arg.epochs, 
                    game=arg.game, 
                    game_lambda=arg.game_lambda, 
                    l2=arg.l2, 
                    recalculate=arg.recalculate)
        print(feature_map.shape)
        plot_tsne(feature_map, label, game_loss, arg.game)
    else:
        res = read_dataset(arg.path, recalculate=arg.recalculate)
        _, _, test_set, _, _, test_label, _,_,_ = res
        predict_model(test_set, test_label, l2=arg.l2, cp_path=arg.checkpoint)
       
    
    


