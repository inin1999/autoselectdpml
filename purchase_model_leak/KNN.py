import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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
        f.write('KNN  ')
        f.write(data_name)
        f.write(' ')
        f.write(ss)
        f.write('\n')

situation_name = ['feature16_zero','feature37_zero','feature16_not_zero','feature37_not_zero','feature16_not_zero_37_not_zero','feature16_not_zero_37_zero','feature16_zero_37_not_zero','feature16_zero_37_zero']

for i in  range(len(situation_name)):
    print(situation_name[i])
    INPUT_PATH = '../data/purchase/9_35_100/'+situation_name[i]+'.csv'
    LABEL_PATH = '../data/purchase/9_35_100/'+situation_name[i]+'_label.csv'
    MODEL_PATH = 'model/'+situation_name[i]+'/KNN'

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

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    print('train_X: ',train_X.shape)
    print('train_Y: ',train_Y.shape)
    print('test_X: ',test_X.shape)
    print('test_Y: ',test_Y.shape)
    
    #train model
    start_time = time.time()
    model = KNeighborsRegressor()
    model.fit(train_X, train_Y)
    # model.fit(train_X, train_Y.ravel())
    end_time = time.time()
    count_time(situation_name[i],'train model',start_time,end_time)
    
    #save model
    joblib.dump(model, MODEL_PATH)
    
    #compute predict a data run time
    start_time = time.time()
    tmp = []
    tmp.append(test_X[0])
    predictions_val = model.predict(tmp)
    end_time = time.time()
    count_time(situation_name[i],'predict a data',start_time,end_time)
    
    # # compute validation scores
    # tmp = []
    # for n in range(100):
    #     tmp.append(test_X[n])
    # print('tmp len: ',len(tmp))
    # predictions_val = model.predict(tmp)
    # count = 0
    # count1 = 0
    # count2 = 0
    # for n in range(len(tmp)):
    #     print('predictions_val[n][0]:',round(predictions_val[n][0]))
    #     if predictions_val[n][0]==test_Y[n][0]:
    #         if predictions_val[n][1]==test_Y[n][1]:
    #             count = count+1
    #     if round(predictions_val[n][0])==test_Y[n][0]:
    #         count1 = count1+1
    #     if predictions_val[n][1]==test_Y[n][1]:
    #         count2 = count2+1

    # print('count: ',count)
    # print('count1: ',count1)
    # print('count2: ',count2)
    # accuracy = float(count/len(test_X))
    # print("accuracy: ",accuracy)
    
    # path = 'record.txt'
    # with open(path, 'a') as f:
    #     f.write('  accuracy: ')
    #     f.write(str(accuracy))
    #     f.write('\n')