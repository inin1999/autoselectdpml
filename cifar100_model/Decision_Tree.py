import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import joblib

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
        f.write('Decision_Tree  ')
        f.write(ss)
        f.write('\n')

INPUT_PATH = '../data/cifar100/9_14_50/all.csv'
LABEL_PATH = '../data/cifar100/9_14_50/all_label.csv'
MODEL_PATH = 'model/Decision_Tree'
CLASS_NUM = 9

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
model = tree.DecisionTreeClassifier()
model.fit(train_X, train_Y)
end_time = time.time()
count_time('train model',start_time,end_time)

#save model
joblib.dump(model, MODEL_PATH)

#compute predict a data run time
start_time = time.time()
tmp = []
tmp.append(test_X[0])
predictions_val = model.predict(tmp)
end_time = time.time()
count_time('predict a data',start_time,end_time)

#compute validation scores
predictions_val = model.predict(test_X)
accuracy = metrics.accuracy_score(test_Y, predictions_val)
print("accuracy: ",accuracy)

path = 'record.txt'
with open(path, 'a') as f:
    f.write('  accuracy: ')
    f.write(str(accuracy))
    f.write('\n')