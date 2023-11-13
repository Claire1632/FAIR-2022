from itertools import count
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data.common.change_rate_data import change_rate_data
# from common import change_rate_data
# def load_data(test_size):
#     data = pd.read_csv('./data/datasets/Vertebral_column.csv')
#     diag_map = {'Abnormal': -1.0, 'Normal': 1.0}
#     data['Label class'] = data['Label class'].map(diag_map)
#     X = data.values[:, 0:-1]
#     Y = data.values[:, 6]
#     X_train, X_test, y_train, y_test = tts(X, Y, test_size=test_size, random_state=42,stratify=Y)
#     return X_train,y_train, X_test, y_test

def load_data(new_rate):
    data = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/FUZZY SVM/CODE/07_04_2022/fuzzy_svm/data/datasets/Vertebral_column.csv')
    diag_map = {'Abnormal': -1.0, 'Normal': 1.0}
    data['Label class'] = data['Label class'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 6]
    # X = data.drop(['Label class'], axis=1)
    # y = data['Label class']
    X, y = change_rate_data(X, y , new_rate = new_rate)

    return X,y
    
    # X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    # #Scalling Data
    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    # return X_train,y_train, X_test, y_test


# X_train,y_train, X_test, y_test = load_data(0.2,1/3)
# X, y = load_data(1/3)
# print(X)
# print(X_train.shape)
# print(X_test.shape)

