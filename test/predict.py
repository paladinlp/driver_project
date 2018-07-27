import numpy as np
import h5py
import time
import pandas as pd



# fileNamePathList = [
#                     'last_h5_2018-06-15-06-20-00_p2_2_last_model.hdf5',
#                     'last_h5_2018-06-15-06-20-00_p2_1_last_model.hdf5',
#                     'InceptionResNetV2_2018-06-13-06-25-50.h5',
#                     'InceptionResNetV2_2018-06-13-08-39-33.h5',
#                     'last_h5_2018-06-25-09-20-42_p3_4_last_model.hdf5',
#                     'last_h5_2018-06-25-09-20-42_p3_2_last_model.hdf5',
#                     'best_h5_2018-06-25-09-20-42_p3_3_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-25-09-20-42_p3_1_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-11-46-12_1_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-11-46-12_2_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-11-46-12_3_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-11-46-12_4_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-11-46-12_6_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-06-01-02_p3_1_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-06-01-02_p3_2_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-06-01-02_p3_6_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-06-01-02_p3_7_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-06-01-02_p3_8_weights_best_InceptionResNetV2.hdf5',
#                     'best_h5_2018-06-15-06-01-02_p3_10_weights_best_InceptionResNetV2.hdf5',
#                     'last_h5_2018-06-15-11-46-12_1_last_model.hdf5'
#                     ]

fileNamePathList = ['Xception.h5']

fileNamePathList = ['h5/'+file for file in fileNamePathList]


X_test = []
for file in fileNamePathList:
    with h5py.File(file, 'r') as h:
        X_test.append(np.array(h['test']))
Y_predict = np.mean([test for test in X_test], axis=0)


def creat_predict(file_name_path_list):
    X_test = []
    for file in fileNamePathList:
        with h5py.File(file, 'r') as h:
            X_test.append(np.array(h['test']))
    X_test = np.array(X_test)
    X_test = X_test.transpose((1, 0, 2))

    def my_func(x):
        arg_max_num = np.argmax(np.bincount([np.argmax(line) for line in x]))
        idx = np.where(np.argmax(x, axis=1) == arg_max_num)
        x = x[idx]
        data = x[x[:, arg_max_num].argsort()]
        return data[-1]

    Y_predict = []
    for item in X_test:
        Y_predict.append(my_func(item))
    return np.array(Y_predict)





# Y_predict = creat_predict(fileNamePathList)


pd_win1 = pd.read_csv('C:/Users\lp\Desktop\csv/last_2018-06-25-15-23-13_win1.csv')
pd_title = (pd_win1['img'])

def creat_csv1(Y_predict,title):
    import time
    fileTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # 写入CSV文件
    df_predict = pd.DataFrame(Y_predict, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df_name_predict = title
    df_predict.insert(0, 'img', df_name_predict)
    fileName = 'C:\\Users\lp\Desktop\csv\\' + fileTime + '_win2' + '.csv'
    fileName1 = 'csv/' + fileTime + '_' + 'all.csv'
    df_predict.to_csv(fileName1, index=False, float_format='%.17f')
    print('****************************************************')
    print('CSV已经生成完毕，名称为：%s' % (fileName1))
    return fileName
fileName = creat_csv1(Y_predict, pd_title)





