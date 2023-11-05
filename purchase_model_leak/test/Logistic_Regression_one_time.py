import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import time
import joblib

print('[Logistic_Regression]')

data_size = 4000

def count_time(start,end):
    temp = end-start
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    # print('Run time --- %d:%d:%d ---' %(hours,minutes,seconds))
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(int(seconds*1000)/1000)
    ss = 'Predict time ---  '+str(hours)+' : '+str(minutes)+' : '+str(seconds)+'  ---'
    return ss

Data_Type = 'leak'
Data_Name = ['feature16_zero','feature16_not_zero','feature37_zero','feature37_not_zero','feature16_zero_37_zero','feature16_not_zero_37_not_zero','feature16_not_zero_37_zero','feature16_zero_37_not_zero']
Data_Filter = ['just_zero','just_not_zero','just_zero','just_not_zero','zero_and_zero','not_zero_and_not_zero','not_zero_and_zero','zero_and_not_zero']
TrainX_Feature = [[16,37],[16,37],[37,37],[37,37],[16,37],[16,37],[16,37],[16,37]]
EXPECT = [[0.08452,0.15884,-0.02948,0.02320,0.03450,0.12580,0],[0.09749,0.15670,-0.01904,0.0752,0.0797,0.0697,0],[0.12224,0.2053,0.00466,0.096,0.0972,0.159,0],[0.09749,0.15670,-0.01904,0.0752,0.0797,0.1138,0],[0.17599,0.27526,0.00094,0.0502,0.138,0.2495,0],[0.2547,0.29204,0.0,0.284,0.2665,0.2602,0],[0.15172,0.23492,-0.00928,0.1008,0.1273,0.148,0],[0.09629,0.18118,-0.0041,0.0383,0.0848,0.062,0]]

for n in range(len(EXPECT)):
    data_type = Data_Type
    data_name = Data_Name[n]
    data_filter = Data_Filter[n]
    print(data_type+' '+data_name)

    trainX_feature = TrainX_Feature[n]
    expect = EXPECT[n]
    
    for run in range(len(expect)):
        Expected = expect[run]
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
        train_y = np.array(train_y)
        train_x = train_x.tolist()

        # #2d to 1d to one-hot
        # train_y = train_y.flatten()
        # train_y = np.eye(9, dtype=np.uint8)[train_y]

        for i in range(len(train_x)):
            train_x[i].insert(0,Expected)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


        MODEL_PATH1 = '../model/'+data_name+'/Logistic_Regression_budget'  #training method 1
        MODEL_PATH2 = '../model/'+data_name+'/Logistic_Regression_acc'
        # MODEL_PATH1 = '../model/8_Situations/Logistic_Regression_budget' #training method 2
        # MODEL_PATH2 = '../model/8_Situations/Logistic_Regression_acc'

        loaded_budget_model = joblib.load(MODEL_PATH1)
        loaded_acc_model = joblib.load(MODEL_PATH2)
        start_time = time.time()
        budget_result = loaded_budget_model.predict(train_x)
        acc_result = loaded_acc_model.predict(train_x)
        end_time = time.time()
        RunTime = count_time(start_time,end_time)
        budget_result = budget_result.tolist()
        acc_result = acc_result.tolist()

        # print(budget_result)
        # print(acc_result)

        budget = []
        for i in range(len(train_x)):
            if budget_result[i] not in budget:
                budget.append(budget_result[i])
        print('max label: ',max(budget),'min label: ',min(budget))

        budget = []
        for i in range(len(train_x)):
            if budget_result[i] not in budget:
                budget.append(budget_result[i])

        total = 0
        for i in range(len(budget)):
            acc = 0.0
            count = 0
            for j in range(len(budget_result)):
                if budget_result[j]==budget[i]:
                    acc = acc+acc_result[j]
                    count = count+1
            acc = float(acc/(count*100))

            if budget[i]==0:
                predicted_budget = '0.01' # budget
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