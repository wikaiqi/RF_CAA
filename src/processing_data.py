#!/usr/bin/env python

import os
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


def process_datafile(path, printinfo=False):
    '''process data file'''
    df_file = pd.read_csv(Path(path, 'file_locator.csv'), names=['file', 'label'])
    df_file['filename']=df_file['file'].apply(lambda x: x.split('/')[-1])
    df_file['subject_id']=df_file['filename'].apply(lambda x: x.split('_')[1])
    df_file['session_id']=df_file['filename'].apply(lambda x: int(x.split('_')[2].replace('sess', '')))
    df_file['record_id']=df_file['filename'].apply(lambda x: int(x.split('_')[3].replace('.txt', '')))
    
    if printinfo:
        df_s1 = df_file[df_file['subject_id']=='s1'].copy()
        df_s2 = df_file[df_file['subject_id']=='s2'].copy()
        #print out df_s1 info
        print("number of recordings for s1: ",df_s1['file'].shape[0])
        print("number of sessions for s1: ", len(df_s1['session_id'].unique()))
        print("number of F sessions for s1: ", len(df_s1[df_s1['label']==0]['session_id'].unique()))
        print("number of MW sessions for s1: ", len(df_s1[df_s1['label']==1]['session_id'].unique()))
        print("number of F recording for s1: ", df_s1[df_s1['label']==0].shape[0])
        print("number of MW recording for s1: ", df_s1[df_s1['label']==1].shape[0])
        print("")
        print("number of recordings for s2: ",df_s2['file'].shape[0])
        print("number of sessions for s2: ", len(df_s2['session_id'].unique()))
        print("number of F sessions for s2: ", len(df_s2[df_s2['label']==0]['session_id'].unique()))
        print("number of MW sessions for s2: ", len(df_s2[df_s2['label']==1]['session_id'].unique()))

    return df_file

def read_txt(file):
    '''read txt file and return raw sensor data'''
    data = []
    with open(file) as fp:
        line1 = fp.readline()
        line2 = fp.readline().split(',')
        data = [float(x) for x in line2]

    return data

def get_powerspectrum(file):
    '''get power spectrum '''
    label = 0 if file.split('/')[-1].split('.')[0].split('_')[0]=='F' else 1
    data = read_txt(file)
    powerSpectrum, freq, time, imageAxis = plt.specgram(data, Fs=1024, noverlap=416, NFFT=450)
    powerSpectrum = powerSpectrum[:224, :224]
    return powerSpectrum, label

if __name__=='__main__':
    path = 'data'
    df_file = process_datafile(path)
    file = df_file['file'].values[0]
    powerSpectrum, label = get_powerspectrum(file) 

    print("spectrum size: {} -- label -- {} -- file: {}".format(powerSpectrum.shape, label, file))
    






