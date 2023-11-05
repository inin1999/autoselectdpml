import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

import time
def count_time(output_str,start,end):
    temp = end-start
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(int(seconds*1000000)/1000000)
    ss = output_str+' run time --- '+str(hours)+' : '+str(minutes)+' : '+str(seconds)+' ---'
    print(ss)
    
    path = 'record.txt'
    with open(path, 'a') as f:
        f.write('CNN  ')
        f.write(ss)
        f.write('\n')

INPUT_PATH = '../data/cifar100/9_14_50/all.csv'
LABEL_PATH = '../data/cifar100/9_14_50/all_label.csv'
MODEL_PATH = 'model/CNN'
CLASS_NUM = 9

chunk_size = 10000
data_list = []
for chunk in pd.read_csv(INPUT_PATH,header=None,index_col=False,chunksize=chunk_size):
    data_list.append(chunk)
Input = pd.concat(data_list, axis=0)
print('X: ',Input.shape)


data_list = []
for chunk in pd.read_csv(LABEL_PATH,header=None,index_col=False,chunksize=chunk_size):
    data_list.append(chunk)
Label = pd.concat(data_list, axis=0)
print('Y: ',Label.shape)

Input = np.array(Input)
Label = np.array(Label)

#Convert to three dimensions
Input = Input.reshape(Input.shape[0], Input.shape[1], 1)/np.max(Input)
print('Input num: ',len(Input))
print('Label num: ',len(Label))

#Modeling
start_time = time.time()
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=CLASS_NUM, activation='softmax'))
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])
history = model.fit(Input, Label, batch_size=512, epochs=150, validation_split=0.1,shuffle=True)
end_time = time.time()
count_time('train model',start_time,end_time)

#save model
tf.saved_model.save(model, MODEL_PATH)

#plot result
def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['accuracy'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    ss = 'cnn_result/result.png'
    plt.savefig(ss,bbox_inches = 'tight')
    # plt.show() 

plot_loss_accuracy(history)