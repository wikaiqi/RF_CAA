
import os
import numpy as np
from pathlib import Path
from .processing_data import process_datafile, get_powerspectrum

def read_dataset(path, train_frac=0.8, recalculate=False):
    '''processing dataset. By default, use 80% of data of as training set, 
        10% validation set, and 10% test set.
        Generate spectrogram data and save in data/spectrogram.
        Set 'recalculate = false' to avoid regenerate the spectrogram every time.
    '''
    df_file = process_datafile(path)
    
    if not os.path.exists(Path(path, 'spectrograms')):
        os.makedirs(Path(path, 'spectrograms'))
    dataset_path = Path(path, 'spectrograms', 'spectrum_data.npy')
    label_path = Path(path, 'spectrograms', 'label.npy')
    sid_path = Path(path, 'spectrograms', 'sid.npy')

    if not dataset_path.is_file() or recalculate:
        label_list, file_list, data_list, subject_id_list = [], [], [], []
        for file in  df_file['file'].values:
            powerSpectrum, label = get_powerspectrum(file)  # return 224*224 time-freq spectrogram
            print("\r file: {}          ".format(file), end='')
            label_list.append(label)
            file_list.append(file)
            data_list.append(powerSpectrum)
        subject_id_list = [int(x.replace('s',''))-1 for x in df_file['subject_id'].values]

        print("")
        dataset, labels, sid= np.array(data_list), np.array(label_list), np.array(subject_id_list)
        dataset = np.expand_dims(dataset, axis=3)
        

        np.save(dataset_path, dataset)
        np.save(label_path, labels)
        np.save(sid_path, sid)

    else:
        dataset = np.load(dataset_path)
        labels = np.load(label_path)
        sid = np.load(sid_path)

    n_train = int(dataset.shape[0]*train_frac)
    n_val = (dataset.shape[0] - n_train)//2
    labels = np.expand_dims(labels, axis=1)
    sid = np.expand_dims(sid,  axis=1)
    train_set, val_set, test_set= dataset[:n_train, :, :, :], dataset[n_train:n_train+n_val, :, :, :], dataset[n_train+n_val:, :, :, :]
    train_label, val_label, test_label = labels[:n_train, :], labels[n_train:n_train+n_val,:],labels[n_train+n_val:, :]
    train_sid, val_sid, test_sid = sid[:n_train, :], sid[n_train:n_train+n_val,:],sid[n_train+n_val:, :]


    print("total number samples: ", dataset.shape[0])
    print("training dataset size: ",train_set.shape)
    print("train set class = 1, n = ", np.sum(train_label))
    print("train set class = 0, n = ", train_set.shape[0]-np.sum(train_label))
    print("train set sid = 1, n = ", np.sum(train_sid))
    print("train set sid = 0, n = ", train_set.shape[0]-np.sum(train_sid))

    print("vall     dataset size: ", val_set.shape)
    print("val set class = 1, n = ", np.sum(val_label))
    print("val set class = 0, n = ", val_set.shape[0]-np.sum(val_label))
    print("val set sid = 1, n = ", np.sum(val_sid))
    print("val set sid = 0, n = ", val_set.shape[0]-np.sum(val_sid))

    print("test     dataset size: ", test_set.shape)
    print("test set class = 1, n = ", np.sum(test_label))
    print("test set class = 0, n = ", test_set.shape[0]-np.sum(test_label))
    print("test set sid = 1, n = ", np.sum(test_sid))
    print("test set sid = 0, n = ", test_set.shape[0]-np.sum(test_sid))
    
    return train_set, val_set, test_set, train_label, val_label, test_label, train_sid, val_sid, test_sid
