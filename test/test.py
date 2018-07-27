import h5py
import numpy as np
from sklearn.utils import shuffle

np.random.seed(2017)



def data_process(*args):
    X_train = []
    X_valid = []
    X_test = []

    # 读取H5文件数据
    for fileNamePath in args:
        with h5py.File(fileNamePath, 'r') as h:
            X_train.append(np.array(h['train']))
            X_valid.append(np.array(h['valid']))
            y_train = np.array(h['train_label'])
            y_valid = np.array(h['valid_label'])
            X_test.append(np.array(h['test']))

    # 融合模型数据
    X_train = np.concatenate(X_train, axis=1)
    X_valid = np.concatenate(X_valid, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    # 转换为独热编码
    n_values = np.max(y_train) + 1
    y_train = np.eye(n_values)[y_train]
    y_valid = np.eye(n_values)[y_valid]

    # 打乱数据
    X_train, y_train = shuffle(X_train, y_train)

    return X_train, X_valid, X_test, y_train, y_valid

X_train, X_valid, X_test, y_train, y_valid = data_process('gap_model_inceptionResNetV2.h5')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import  log_loss, make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV
from  sklearn.tree import DecisionTreeClassifier

def ado_net(X_train, y_train, X_valid, y_valid):
    # 创建分类器
    clf = AdaBoostClassifier(random_state=0)

    y_train = np.argmax(y_train, axis=1)
    y_valid = np.argmax(y_valid, axis=1)

    # scorer = make_scorer(log_loss)
    # 设置网格调整参数
    parameters_ado = {
        'n_estimators': [50, 100, 200, 1000],
        'learning_rate': [1.0, 0.5, 0.2, 0.1, 0.01],
    }
    # parameters = {'max_depth': [3, 5, 10, 20, 50],
    #               'criterion': ['gini', 'entropy'],
    #               'min_samples_leaf': [1, 3, 5],
    #               'min_samples_split': [5, 10],
    #               'max_leaf_nodes': [None, 5, 10, 100, 500]}
    # 创建K折对象
    cross_validator = KFold(n_splits=2)

    # 在分类器上使用网格搜索
    grid_obj = GridSearchCV(clf, parameters_ado, cv=cross_validator, scoring='neg_log_loss')

    grid_obj.fit(X_train, y_train)


    acc = accuracy_score(grid_obj.predict(X_valid), y_valid)
    print(acc)

ado_net(X_train, y_train, X_valid, y_valid)