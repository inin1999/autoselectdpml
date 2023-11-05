import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
import time

print('[CNN]')

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

expect = [0.00921, 0.09694, -0.00668, 0.005, 0.0026, 0.0119, 0.0]
data_size = 10000

for n in range(len(expect)):
    Expected = expect[n]
    print('expect: ',Expected)

    target_data = '../../original_data/cifar100/target_data/target_data.npz'
    with np.load(target_data) as f:
        train_x, train_y, _, _ = [f['arr_%d' % i] for i in range(len(f.files))]

    print('train_x: ',len(train_x))
    # print('train_y: ',len(train_y))

    train_x = np.array(train_x, dtype=np.float32)
    # train_y = np.array(train_y, dtype=np.int32)
    train_x = train_x.tolist()
    # train_y = train_y.tolist()

    for i in range(len(train_x)):
        train_x[i].insert(0,Expected)

    train_x = np.array(train_x)

    #Convert to three dimensions
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)/np.max(train_x)
    train_x = np.expand_dims(train_x, axis=1)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def evaluate(model):
        list = []
        # acc = 0
        for j in range(len(train_x)):
            predictions = model([train_x[j]])
            predictions[0] = tf.argmax(predictions[0], 1)
            list.append(predictions)
            # if tf.argmax(predictions, 1)==train_y[j]:
            #     acc = acc+1
        # print('acc: ', acc*1.0/(len(train_x)))
        return list

    
    MODEL_PATH = '../model/CNN'
    
    model = tf.saved_model.load(MODEL_PATH)
    result = []
    start_time = time.time()
    result = evaluate(model)
    end_time = time.time()
    RunTime = count_time(start_time,end_time)
    # print(result[0][0])
    # print(result[0][1])
    # print(result[1][0])
    # print(result[1][1])

    budget = []
    for i in range(len(train_x)):
        if result[i][0] not in budget:
            budget.append(result[i][0])
    print('max label: ',max(budget),'min label: ',min(budget))

    budget = []
    for i in range(len(train_x)):
        if np.round(result[i][0].numpy()) not in budget:
            budget.append(np.round(result[i][0].numpy()))

    total = 0
    for i in range(len(budget)):
        acc = 0.0
        count = 0
        for j in range(len(result)):
            if np.round(result[j][0].numpy())==budget[i]:
                acc = acc+result[j][1]
                count = count+1
        acc = float(acc/count)

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