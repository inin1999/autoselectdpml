import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

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
        f.write('Logistic_Regression  ')
        f.write(data_name)
        f.write(' ')
        f.write(ss)
        f.write('\n')

situation_name = ['feature16_zero','feature37_zero','feature16_not_zero','feature37_not_zero','feature16_not_zero_37_not_zero','feature16_not_zero_37_zero','feature16_zero_37_not_zero','feature16_zero_37_zero']

for i in  range(len(situation_name)):
    print(situation_name[i])
    INPUT_PATH = '../data/purchase/9_35_100/'+situation_name[i]+'.csv'
    LABEL_PATH = '../data/purchase/9_35_100/'+situation_name[i]+'_label.csv'
    MODEL_PATH = 'model/'+situation_name[i]+'/Logistic_Regression'
    CLASS_NUM = 9 #types of privacy budget
  
    #read data
    chunk_size = 10000
    data_list = []
    for chunk in pd.read_csv(INPUT_PATH,header=None,index_col=False,chunksize=chunk_size):
        data_list.append(chunk)
    X = pd.concat(data_list, axis=0)
    print('X: ',X.shape)
    
    
    data_list = []
    for chunk in pd.read_csv(LABEL_PATH,header=None,index_col=False,chunksize=chunk_size):
        data_list.append(chunk)
    Y = pd.concat(data_list, axis=0)
    print('Y: ',Y.shape)

    train_X, test_X, train_Y, test_Y = train_test_split(X,Y, random_state=777, train_size=0.9, shuffle=True, stratify=None)

    x_train = np.array(train_X)
    y_train = np.array(train_Y)
    x_test = np.array(test_X)
    y_test = np.array(test_Y)
    
    #2d to 1d to one-hot
    y_train = y_train.flatten()
    y_train = np.eye(9, dtype=np.uint8)[y_train]
    y_test = y_test.flatten()
    y_test = np.eye(9, dtype=np.uint8)[y_test]

    # Define the model
    model = Sequential([
        Dense(9, activation='softmax', input_shape=(102,))
    ])

    # Compile the model with appropriate loss function, optimizer and metrics
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # Train the model
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=300, batch_size=128)
    end_time = time.time()
    count_time(situation_name[i],'train model with gpu',start_time,end_time)

    # save model
    tf.saved_model.save(model, MODEL_PATH)

    #plot result
    def plot_loss_accuracy(history):
        historydf = pd.DataFrame(history.history, index=history.epoch)
        plt.figure(figsize=(8, 6))
        historydf.plot(ylim=(0, max(1, historydf.values.max())))
        print(history.history.keys())
        loss = history.history['loss'][-1]
        acc = history.history['acc'][-1]
        plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
        ss = 'lr_result/'+situation_name[i]+'_result.png'
        plt.savefig(ss,bbox_inches = 'tight')

    plot_loss_accuracy(history)

    #evaluate the model on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    path = 'record.txt'
    with open(path, 'a') as f:
        f.write('  accuracy: ')
        f.write(str(accuracy))
        f.write('\n')