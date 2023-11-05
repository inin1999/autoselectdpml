import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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

INPUT_PATH = '../data/purchase/8_Situations/9_35_100.csv'
LABEL_PATH = '../data/purchase/8_Situations/9_35_100_label.csv'
MODEL_PATH = 'model/8_Situations/KNN'
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

train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)

#train model
start_time = time.time()
model = KNeighborsClassifier()
model.fit(train_X, train_Y.ravel())
end_time = time.time()
count_time('8_Situations','train model',start_time,end_time)

#save model
joblib.dump(model, MODEL_PATH)

#compute predict a data run time
start_time = time.time()
tmp = []
tmp.append(test_X[0])
predictions_val = model.predict(tmp)
end_time = time.time()
count_time('8_Situations','predict a data',start_time,end_time)

#compute validation scores
predictions_val = model.predict(test_X)
accuracy = metrics.accuracy_score(test_Y, predictions_val)
print("accuracy: ",accuracy)

path = 'record.txt'
with open(path, 'a') as f:
    f.write('  accuracy: ')
    f.write(str(accuracy))
    f.write('\n')