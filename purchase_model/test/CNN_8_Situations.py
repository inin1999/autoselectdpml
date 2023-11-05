import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import time

print('[CNN]')

# 9_35_100 Situation 1
data_name = 'feature16_zero'
data_filter = 'just_zero'
trainX_feature = [16,37]
expect = [[0.69936,0.08452],[0.94220,0.15884],[0.17380,-0.02948],[0.44600,0.02320],[0.57300,0.03450],[0.85600,0.12580],[0.95060,0]]

## 9_35_100 Situation 2
# data_name = 'feature16_not_zero'
# data_filter = 'just_not_zero'
# trainX_feature = [16,37]
# expect = [[0.68781,0.09749],[0.91100,0.15670],[0.21480,-0.01904],[0.446,0.0752],[0.522,0.0797],[0.746,0.0697],[0.90190,0]]

## 9_35_100 Situation 3
# data_name = 'feature37_zero'
# data_filter = 'just_zero'
# trainX_feature = [37,37]
# expect = [[0.68413,0.12224],[0.935,0.2053],[0.174,0.00466],[0.379,0.096],[0.636,0.0972],[0.803,0.159],[0.9309,0]]	

## 9_35_100 Situation 4
# data_name = 'feature37_not_zero'
# data_filter = 'just_not_zero'
# trainX_feature = [37,37]
# expect = [[0.68781,0.09749],[0.91100,0.15670],[0.21480,-0.01904],[0.446,0.0752],[0.522,0.0797],[0.708,0.1138],[0.8988,0]]	

## 9_35_100 Situation 5
# data_name = 'feature16_zero_37_zero'
# data_filter = 'zero_and_zero'
# trainX_feature = [16,37]
# expect = [[0.6577,0.17599],[0.9252,0.27526],[0.1298,0.00094],[0.323,0.0502],[0.576,0.138],[0.852,0.2495],[0.92670,0]]	

## 9_35_100 Situation 6
# data_name = 'feature16_not_zero_37_not_zero'
# data_filter = 'not_zero_and_not_zero'
# trainX_feature = [16,37]
# expect = [[0.64511,0.2547],[0.8628,0.29204],[0.2628,0.0],[0.42,0.284],[0.514,0.2665],[0.711,0.2602],[0.8438,0]]	

## 9_35_100 Situation 7
# data_name = 'feature16_not_zero_37_zero'
# data_filter = 'not_zero_and_zero'
# trainX_feature = [16,37]
# expect = [[0.67246,0.15172],[0.908,0.23492],[0.1976,-0.00928],[0.451,0.1008],[0.658,0.1273],[0.772,0.148],[0.88950,0]]	

## 9_35_100 Situation 8
# data_name = 'feature16_zero_37_not_zero'
# data_filter = 'zero_and_not_zero'
# trainX_feature = [16,37]
# expect = [[0.68143,0.09629],[0.9162,0.18118],[0.221,-0.0041],[0.424,0.0383],[0.649,0.0848],[0.817,0.062],[0.90590,0]]	

data_type = '8_Situations'
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
    seconds = float(int(seconds*1000)/1000)
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
        train_x[i].insert(0,Expected[0])
        train_x[i].insert(1,Expected[1])

    #Convert to three dimensions
    train_x = np.array(train_x, dtype=np.float32)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)/np.max(train_x)
    # train_y = np.array(train_y, dtype=np.int32)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def evaluate(model):
        list = []
        # acc = 0
        for j in range(len(train_x)):
            predictions = model([train_x[j]])
            tmp = tf.argmax(predictions, 1)
            list.append(tmp)
            # if tf.argmax(predictions, 1)==train_y[j]:
            #     acc = acc+1
        # print('acc: ', acc*1.0/(len(train_x)))

    MODEL_PATH = '../model/8_Situations/CNN'

    model = tf.saved_model.load(MODEL_PATH)
    result = []
    start_time = time.time()
    result = evaluate(model)
    end_time = time.time()
    RunTime = count_time(start_time,end_time)

    print('predict 0.01: ',result.count(0))
    print('predict 0.05: ',result.count(1))
    print('predict 0.1 : ',result.count(2))
    print('predict 0.5 : ',result.count(3))
    print('predict 1   : ',result.count(4))
    print('predict 5   : ',result.count(5))
    print('predict 10  : ',result.count(6))
    print('predict 100 : ',result.count(7))
    print('predict 1000: ',result.count(8))
    print(RunTime)