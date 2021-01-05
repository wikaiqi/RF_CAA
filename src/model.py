import os
import shutil
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, Input, Model
from .resnet import Encoder, Predictor, Discriminator
from .dataset import read_dataset

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def train_model(path, n_epochs, batch_size=32, l2=0.01, game=False,
                game_lambda=1.5, recalculate=False):
    '''train deep learning model by using tensorflow 2.0'''
    res = read_dataset(path, recalculate=recalculate) 
    train_set, val_set, test_set, train_label, val_label, test_label, train_sid, val_sid, test_sid  = res
    BUFFER_SIZE, data_size = train_set.shape[0], train_set.shape[1]
    n_steps = BUFFER_SIZE // batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices((train_set, train_label, train_sid)).shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(batch_size)

    encoder = Encoder(l2=l2)
    predictor = Predictor()
    discriminator = Discriminator()

    #setup learning rate scheduler
    boundaries = [50*n_steps]
    lr_values = [0.1, 0.01]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, lr_values)
    opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate_fn)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    game_loss = tf.keras.metrics.Mean(name='game_loss')
    game_accuracy = tf.keras.metrics.BinaryAccuracy(name='game_accuracy')

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    #@tf.function
    def train_step(input, targets, l2=0.01):
        with tf.GradientTape() as tape:
            enc_output = encoder(input, training = True)
            predictions = predictor(enc_output, training = True)

            loss = loss_obj(targets, predictions)

            variables = encoder.trainable_variables + predictor.trainable_variables
            loss_L2 = tf.add_n([tf.nn.l2_loss(v) for v in variables])*l2
            loss += loss_L2
        
        gradients = tape.gradient(loss, variables)
        opt.apply_gradients(zip(gradients, variables))

        train_loss.update_state(loss)
        train_accuracy.update_state(targets, predictions)

        return loss

    @tf.function
    def game_step(input, targets, targets_s, game_lambda=1.5, l2=0.01):
        '''discriminator trainning'''
        with tf.GradientTape(persistent=True) as tape:
            enc_output = encoder(input, training=True)
            condition_dis = predictor(enc_output, training=False)
            
            pred_loss = loss_obj(targets, condition_dis) 
            pred_variables = encoder.trainable_variables + predictor.trainable_variables
            pred_loss += (tf.add_n([tf.nn.l2_loss(v) for v in pred_variables])*l2)

            predictions = discriminator(enc_output, training=True)
            dis_loss = loss_obj(targets_s, predictions)
            dis_variables = encoder.trainable_variables + discriminator.trainable_variables 
            dis_loss += (tf.add_n([tf.nn.l2_loss(v) for v in dis_variables])*l2)

            total_loss  = pred_loss - game_lambda*dis_loss

        #update encoder
        variables = encoder.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        opt.apply_gradients(zip(gradients, variables))

        
        #update predictor
        variables = predictor.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        opt.apply_gradients(zip(gradients, variables))

        for i in range(5):
            variables = discriminator.trainable_variables
            gradients = tape.gradient(total_loss, variables)
            gradients = [-1*x  for x in gradients]
            opt.apply_gradients(zip(gradients, variables))
        

        game_loss.update_state(total_loss)
        train_loss.update_state(pred_loss)
        train_accuracy.update_state(targets, condition_dis)
        game_accuracy.update_state(targets_s, predictions)  # measure how well the discriminator it is



    @tf.function
    def test_step(input, targets):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        enc_output = encoder(input, training = False)
        predictions = predictor(enc_output, training = False)

        t_loss = loss_obj(targets, predictions)
        variables = encoder.trainable_variables + predictor.trainable_variables
        t_loss_L2 = tf.add_n([tf.nn.l2_loss(v) for v in variables])*0.005
        t_loss += t_loss_L2
        
        test_loss(t_loss)
        test_accuracy(targets, predictions)


    checkpoint = tf.train.Checkpoint(#opt=opt,
                                    encoder=encoder,
                                    predictor=predictor)
    train_acc, test_acc, max_acc, wait_step = [], [], -100.0, 0
    print('----------------------------------------------------------')
    game_loss_list = []
    for epoch in range(n_epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for step, (inp, labels, sid) in enumerate(train_dataset.take(n_steps)):
            
            
            if game:
                game_step(inp, labels, sid, game_lambda=game_lambda, l2=l2)
            else: 
                train_step(inp, labels, l2=l2)
               

        test_step(val_set, val_label)
 
        train_acc.append(train_accuracy.result())
        test_acc.append(test_accuracy.result())

        if max_acc < test_accuracy.result():
            checkpoint.save(file_prefix = 'check_point/check_point.ckpt')

            max_acc = test_accuracy.result()
       
    
        print('\r Epoch {}, train_Loss:{:7.4f},  Train_acc:{:7.4f}, val_Loss:{:7.4f}, val_acc:{:7.4f} max_acc:{:7.4f}'.format(
                epoch+1, train_loss.result(), train_accuracy.result(),
                test_loss.result(), test_accuracy.result(), max_acc), end='', flush=True)
        if game:
            print(', game_loss:{:7.4f},  dis_loss:{:7.4f}   '.format(game_loss.result(), (train_loss.result()-game_loss.result())/game_lambda), end='', flush=True)
            game_loss_list.append(game_loss.result())

    
    #precess final results. retrevial best model from saved checkpoint
    print("")
    print('----------------------------------------------------------')
    checkpoint.restore(tf.train.latest_checkpoint('check_point/'))
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_step(train_set, train_label)
    print("train loss: {:7.4f}, train acc:{:7.4f}".format(test_loss.result(), test_accuracy.result()))

    test_loss.reset_states()
    test_accuracy.reset_states()
    test_step(val_set, val_label)
    print("val  loss: {:7.4f}, val  acc:{:7.4f}".format(test_loss.result(), test_accuracy.result()))

    test_loss.reset_states()
    test_accuracy.reset_states()
    test_step(test_set, test_label)
    print("test loss: {:7.4f}, test acc:{:7.4f}".format(test_loss.result(), test_accuracy.result()))

    test_enc_output = encoder(test_set, training = False)
    feature_map = test_enc_output.numpy()

    print('-----------------------END--------------------------------')

    return feature_map,test_label, game_loss_list

  
def predict_model(test_set, test_label, l2=0.01, cp_path='check_point/'):
    encoder = Encoder(l2=l2)
    predictor = Predictor()
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    checkpoint = tf.train.Checkpoint(encoder=encoder,
                                    predictor=predictor)

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    @tf.function
    def test_step(input, targets):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        enc_output = encoder(input, training = False)
        predictions = predictor(enc_output, training = False)

        t_loss = loss_obj(targets, predictions)
        variables = encoder.trainable_variables + predictor.trainable_variables
        t_loss_L2 = tf.add_n([tf.nn.l2_loss(v) for v in variables])*0.005
        t_loss += t_loss_L2
        
        test_loss(t_loss)
        test_accuracy(targets, predictions)

        return enc_output

    checkpoint.restore(tf.train.latest_checkpoint(cp_path))
    test_loss.reset_states()
    test_accuracy.reset_states()
    enc_output = test_step(test_set, test_label)
    print("Test accuracy:{}".format(test_accuracy.result()))
    feature_map = enc_output.numpy()

    return feature_map,test_label, []
    
