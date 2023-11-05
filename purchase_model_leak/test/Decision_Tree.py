import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import joblib
import time

print('[Decision Tree]')

# 9_35_100 Situation 1
data_name = 'feature16_zero'
data_filter = 'just_zero'
trainX_feature = [16,37]
expect = [0.08452,0.15884,-0.02948,0.02320,0.03450,0.12580,0]

## 9_35_100 Situation 2
# data_name = 'feature16_not_zero'
# data_filter = 'just_not_zero'
# trainX_feature = [16,37]
# expect = [0.09749,0.15670,-0.01904,0.0752,0.0797,0.0697,0]

## 9_35_100 Situation 3
# data_name = 'feature37_zero'
# data_filter = 'just_zero'
# trainX_feature = [37,37]
# expect = [0.12224,0.2053,0.00466,0.096,0.0972,0.159,0]

## 9_35_100 Situation 4
# data_name = 'feature37_not_zero'
# data_filter = 'just_not_zero'
# trainX_feature = [37,37]
# expect = [0.09749,0.15670,-0.01904,0.0752,0.0797,0.1138,0]

## 9_35_100 Situation 5
# data_name = 'feature16_zero_37_zero'
# data_filter = 'zero_and_zero'
# trainX_feature = [16,37]
# expect = [0.17599,0.27526,0.00094,0.0502,0.138,0.2495,0]

## 9_35_100 Situation 6
# data_name = 'feature16_not_zero_37_not_zero'
# data_filter = 'not_zero_and_not_zero'
# trainX_feature = [16,37]
# expect = [0.2547,0.29204,0.0,0.284,0.2665,0.2602,0]

## 9_35_100 Situation 7
# data_name = 'feature16_not_zero_37_zero'
# data_filter = 'not_zero_and_zero'
# trainX_feature = [16,37]
# expect = [0.15172,0.23492,-0.00928,0.1008,0.1273,0.148,0]

## 9_35_100 Situation 8
# data_name = 'feature16_zero_37_not_zero'
# data_filter = 'zero_and_not_zero'
# trainX_feature = [16,37]
# expect = [0.09629,0.18118,-0.0041,0.0383,0.0848,0.062,0]

data_type = '9_35_100'
data_size = 4000
print(data_type+' '+data_name)

def count_time(start,end):
    temp = end-start
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    # print('Run time --- %d:%d:%d ---' %(hours,minutes,seconds))
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(int(seconds*1000000)/1000000)
    ss = 'Predict time ---  '+str(hours)+' : '+str(minutes)+' : '+str(seconds)+'  ---'
    return ss

for n in range(len(expect)):
    Expected = expect[n]
    print('expect: ',Expected)

    target_data = '../../original_data/purchase/target_data/target_data.npz'
    with np.load(target_data) as f:
        train_x, train_y, _, _ = [f['arr_%d' % i] for i in range(len(f.files))]

    print('train_x: ',len(train_x))
    # print('train_y: ',len(train_y))

    tmp1 = []
    # tmp2 = []
    count = 0
    for i in range(len(train_x)):
        if data_filter=='just_zero':
            if train_x[i][[trainX_feature[0]]]==0 and count<data_size*2:
                if count>=data_size:
                    tmp1.append(train_x[i])
                    # tmp2.append(train_y[i])
                count = count+1
        if data_filter=='just_not_zero':
            if train_x[i][[trainX_feature[0]]]!=0 and count<data_size*2:
                if count>=data_size:
                    tmp1.append(train_x[i])
                    # tmp2.append(train_y[i])
                count = count+1
        if data_filter=='zero_and_zero':
            if train_x[i][[trainX_feature[0]]]==0 and train_x[i][[trainX_feature[1]]]==0 and count<data_size*2:
                if count>=data_size:
                    tmp1.append(train_x[i])
                    # tmp2.append(train_y[i])
                count = count+1
        if data_filter=='not_zero_and_zero':
            if train_x[i][[trainX_feature[0]]]!=0 and train_x[i][[trainX_feature[1]]]==0 and count<data_size*2:
                if count>=data_size:
                    tmp1.append(train_x[i])
                    # tmp2.append(train_y[i])
                count = count+1
        if data_filter=='zero_and_not_zero':
            if train_x[i][[trainX_feature[0]]]==0 and train_x[i][[trainX_feature[1]]]!=0 and count<data_size*2:
                if count>=data_size:
                    tmp1.append(train_x[i])
                    # tmp2.append(train_y[i])
                count = count+1
        if data_filter=='not_zero_and_not_zero':
            if train_x[i][[trainX_feature[0]]]!=0 and train_x[i][[trainX_feature[1]]]!=0 and count<data_size*2:
                if count>=data_size:
                    tmp1.append(train_x[i])
                    # tmp2.append(train_y[i])
                count = count+1
        if data_filter=='all':
            if count<data_size*2:
                if count>=data_size:
                    tmp1.append(train_x[i])
                    # tmp2.append(train_y[i])
                count = count+1
    train_x = tmp1
    # train_y = tmp2

    print('train_x: ',len(train_x)) #4000
    # print('train_y: ',len(train_y))

    train_x = np.array(train_x, dtype=np.float32)
    # train_y = np.array(train_y, dtype=np.int32)
    train_x = train_x.tolist()
    # train_y = train_y.tolist()

    for i in range(len(train_x)):
        train_x[i].insert(0,Expected)

    train_x = np.array(train_x, dtype=np.float32)
    # train_y = np.array(train_y, dtype=np.int32)

    MODEL_PATH = '../model/'+data_name+'/Decision_Tree'

    loaded_model = joblib.load(MODEL_PATH)
    start_time = time.time()
    result = loaded_model.predict(train_x)
    end_time = time.time()
    RunTime = count_time(start_time,end_time)
    result = result.tolist()
    
    budget = []
    for i in range(len(train_x)):
        if result[i][0] not in budget:
            budget.append(result[i][0])
    print('max label: ',max(budget),'min label: ',min(budget))

    budget = []
    for i in range(len(train_x)):
        if round(result[i][0]) not in budget:
            budget.append(round(result[i][0]))

    total = 0
    for i in range(len(budget)):
        acc = 0.0
        count = 0
        for j in range(len(result)):
            if round(result[j][0])==budget[i]:
                acc = acc+result[j][1]
                count = count+1
        acc = float(acc/count)

        if budget[i]==0:
            predicted_budget = '0.01'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        elif budget[i]==1:
            predicted_budget = '0.05'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        elif budget[i]==2:
            predicted_budget = '0.1'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        elif budget[i]==3:
            predicted_budget = '0.5'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        elif budget[i]==4:
            predicted_budget = '1'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        elif budget[i]==5:
            predicted_budget = '5'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        elif budget[i]==6:
            predicted_budget = '10'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        elif budget[i]==7:
            predicted_budget = '100'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        elif budget[i]==8:
            predicted_budget = '1000'
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        else:
            predicted_budget = budget[i]
            predicted_budget = 'not in range'+str(predicted_budget)
            print('[label=',budget[i],',budget=',predicted_budget,' ,acc=',acc,']: ',count)
        total = total+count
    print('total: ',total,'\n')