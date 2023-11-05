import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
import joblib

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

INPUT_PATH = '../data/purchase/8_Situations/9_35_100.csv'
LABEL_PATH = '../data/purchase/8_Situations/9_35_100_label.csv'
MODEL_PATH1 = 'model/8_Situations/Logistic_Regression_budget'
MODEL_PATH2 = 'model/8_Situations/Logistic_Regression_acc'

#read data
X = pd.read_csv(INPUT_PATH,header=None,index_col=False)
Y = pd.read_csv(LABEL_PATH,header=None,index_col=False)

train_X, test_X, train_Y, test_Y = train_test_split(X,Y, random_state=777, train_size=0.9, shuffle=True, stratify=None)

train_Y_budget = []
train_Y_acc = []
test_Y_budget = []
test_Y_acc = []

x_train = np.array(train_X)
y_train = np.array(train_Y)
x_test = np.array(test_X)
y_test = np.array(test_Y)

for n in  range(len(y_train)):
    train_Y_budget.append(y_train[n][0])
    train_Y_acc.append(int(y_train[n][1]*10000))
for n in  range(len(y_test)):
    test_Y_budget.append(y_test[n][0])
    test_Y_acc.append(int(y_test[n][1]*10000))

train_Y_budget = np.array(train_Y_budget)
train_Y_acc = np.array(train_Y_acc)
test_Y_budget = np.array(test_Y_budget)
test_Y_acc = np.array(test_Y_acc)
print('train_Y_budget: ',train_Y_budget.shape)
print('train_Y_acc: ',train_Y_acc.shape)
print('test_Y_budget: ',test_Y_budget.shape)
print('test_Y_acc: ',test_Y_acc.shape)

#constructing a Logistic Regression model to predict the accuracy
start_time = time.time()
acc_model = LogisticRegression(max_iter=10000)
# acc_model.fit(x_train, train_Y_acc)
# acc_model = SGDClassifier(loss='log')
# acc_model = LinearRegression()
batch_size = 100000
num_batches = len(x_train) // batch_size
for j in range(num_batches-1):
    start_idx = j * batch_size
    end_idx = (j + 1) * batch_size
    # acc_model.partial_fit(x_train[start_idx:end_idx], train_Y_acc[start_idx:end_idx], classes=np.unique(train_Y_acc))
    acc_model.fit(x_train[start_idx:end_idx], train_Y_acc[start_idx:end_idx])
    # acc_model.partial_fit(x_train[start_idx:end_idx], train_Y_acc[start_idx:end_idx], classes=None)
end_time = time.time()
count_time('8_Situations','train acc model',start_time,end_time)

#constructing Logistic Regression models to predict privacy budgets
start_time = time.time()
budget_model = LogisticRegression()
budget_model.fit(x_train, train_Y_budget)
end_time = time.time()
count_time('8_Situations','train budget model',start_time,end_time)

predicted_budget = budget_model.predict(x_test)
predicted_acc = acc_model.predict(x_test)

# save model
joblib.dump(budget_model, MODEL_PATH1)
joblib.dump(acc_model, MODEL_PATH2)

count1=0 
count2=0    
for n in range(len(x_test)):
    pre_budget = predicted_budget[n]
    pre_acc = predicted_acc[n]
    if pre_budget == test_Y_budget[n]:
        count1=count1+1
    if pre_acc == test_Y_acc[n]:
        count2=count2+1
    # print("result | budget={}, acc={}".format(predicted_budget, predicted_acc))
accuracy1=float(count1/len(x_test))
accuracy2=float(count2/len(x_test))
print('count1: ',count1)
print('count2: ',count2)
print('accuracy | budget=',str(accuracy1),', acc=',str(accuracy2))     