import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D
import matplotlib.pyplot as plt
import time

#calculate training time
def count_time(data_name,output_str,start,end):
    temp = end-start
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    # print('Run time --- %d:%d:%d ---' %(hours,minutes,seconds))
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(int(seconds*1000000)/1000000)
    ss = output_str+' run time --- '+str(hours)+' : '+str(minutes)+' : '+str(seconds)+' ---'
    print(ss)
    
    path = 'record.txt'
    with open(path, 'a') as f:
        f.write('CNN  ')
        f.write(data_name)
        f.write(' ')
        f.write(ss)
        f.write('\n')

INPUT_PATH = '../data/purchase/8_Situations/9_35_100.csv'
LABEL_PATH = '../data/purchase/8_Situations/9_35_100_label.csv'
MODEL_PATH = 'model/8_Situations/CNN'
CLASS_NUM = 9 #types of privacy budget

#read data
chunk_size = 10000
data_list = []
for chunk in pd.read_csv(INPUT_PATH,header=None,index_col=False,chunksize=chunk_size):
    data_list.append(chunk)
input = pd.concat(data_list, axis=0)
print('X: ',input.shape)

data_list = []
for chunk in pd.read_csv(LABEL_PATH,header=None,index_col=False,chunksize=chunk_size):
    data_list.append(chunk)
Label = pd.concat(data_list, axis=0)
print('Y: ',Label.shape)

Label_budget=[]
Label_acc=[]
for n in range(len(Label)):
    Label_budget.append(Label[0][n])
    Label_acc.append(Label[1][n])
Label_budget = np.array(Label_budget)
Label_acc = np.array(Label_acc)
#2d to 1d to one-hot
# Label_budget.astype('int64')
Label_budget = Label_budget.flatten()
Label_budget = Label_budget.astype('int64')
Label_budget = np.eye(9, dtype=np.uint8)[Label_budget]

input = np.array(input)

#Convert to three dimensions
input = input.reshape(input.shape[0], input.shape[1], 1)/np.max(input)
input = np.expand_dims(input, axis=1)
print('Input num: ',len(input))
print('Label_budget num: ',len(Label_budget))
print('Label_acc num: ',len(Label_acc))

#confirm tensorFlow to see the available GPU installations
tf.config.list_physical_devices('GPU')
#specify the GPU to use (assuming only one GPU)
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
#or use the following code to limit tensorFlow to only run on certain GPUs
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
#confirm the visible devices that tensorFlow can see.
tf.config.list_logical_devices('GPU')

#modeling
start_time = time.time()
input_shape = input[0].shape
print(input_shape)
inputs = Input(shape=input_shape)
conv1 = Conv1D(32, kernel_size=3, activation='relu')(inputs)
maxpool1 = MaxPooling2D(pool_size=1)(conv1)
conv2 = Conv1D(64, kernel_size=3, activation='relu')(maxpool1)
maxpool2 = MaxPooling2D(pool_size=1)(conv2)
flatten = Flatten()(maxpool2)
dense1 = Dense(128, activation='relu')(flatten)
budget_output = Dense(units=9, activation='softmax', name='budget')(dense1)
acc_output = Dense(1, activation='sigmoid', name='acc')(dense1)

model = Model(inputs=inputs, outputs=[budget_output, acc_output])
model.compile(loss={'budget': 'categorical_crossentropy', 'acc': 'binary_crossentropy'},
            optimizer='adam',metrics={'budget': 'categorical_accuracy', 'acc': 'accuracy'})

# history = model.fit(input, {'budget': Label_budget, 'acc': Label_acc}, epochs=250, batch_size=512)
history = model.fit(input, dict(budget=Label_budget, acc=Label_acc), epochs=250, batch_size=512)
end_time = time.time()
count_time('8_Situations','train model',start_time,end_time)

#save model
tf.saved_model.save(model, MODEL_PATH)

#plot result
def plot_loss_budget_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['budget_loss'][-1]
    acc = history.history['budget_categorical_accuracy'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    ss = 'cnn_result/8_Situations_budget_result.png'
    plt.savefig(ss,bbox_inches = 'tight')
    # plt.show() 

def plot_loss_acc_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['acc_loss'][-1]
    acc = history.history['acc_accuracy'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    ss = 'cnn_result/8_Situations_acc_result.png'
    plt.savefig(ss,bbox_inches = 'tight')
    # plt.show() 

plot_loss_budget_accuracy(history)
plot_loss_acc_accuracy(history)