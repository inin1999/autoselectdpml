import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

MERGE_DATA_INPUT_PATH = 'data/purchase/8_Situations/9_35_100.csv'
MERGE_DATA_LABEL_PATH = 'data/9_35_100_label.csv'

name = ['feature16_zero','feature37_zero','feature16_not_zero','feature37_not_zero','feature16_not_zero_37_not_zero','feature16_not_zero_37_zero','feature16_zero_37_not_zero','feature16_zero_37_zero']
for i in range(8):
    print(name[i])
    input_path = 'data/purchase/9_35_100/'+ name[i] +'.csv'
    label_path = 'data/purchase/9_35_100/'+ name[i] +'_label.csv'

    X = pd.read_csv(input_path,header=None,index_col=False)
    Y = pd.read_csv(label_path,header=None,index_col=False)

    Input = np.array(X)
    Label = np.array(Y)
    print('input: ',len(Input))
    print('label: ',len(Label))

    with open(MERGE_DATA_INPUT_PATH, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Input)        # 寫入二維表格
        csvfile.close()
    with open(MERGE_DATA_LABEL_PATH, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Label)        # 寫入二維表格
        csvfile.close()
    read = pd.read_csv(MERGE_DATA_INPUT_PATH,header=None,index_col=False)
    print('input.csv: ',len(read))
    read = pd.read_csv(MERGE_DATA_LABEL_PATH,header=None,index_col=False)
    print('label.csv: ',len(read))
print('Done.')
