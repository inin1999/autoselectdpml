import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import joblib
import time

print('[Random Forest]')

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

expect = [[0.04891,0.00921],[0.13080,0.09694],[0.00800,-0.00668],[0.01,0.005],[0.078000002,0.0026],[0.118000001,0.0119],[0.165400001,0.0]]
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
        train_x[i].insert(0,Expected[0])
        train_x[i].insert(1,Expected[1])

    train_x = np.array(train_x, dtype=np.float32)
    # train_y = np.array(train_y, dtype=np.int32)

    MODEL_PATH = '../model/Random_Forest'

    loaded_model = joblib.load(MODEL_PATH)
    start_time = time.time()
    result = loaded_model.predict(train_x)
    end_time = time.time()
    RunTime = count_time(start_time,end_time)
    result = result.tolist()

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