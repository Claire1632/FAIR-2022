import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from common.change_rate_data import change_rate_data

def load_data(test_size):
    dataset = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/FUZZY SVM/CODE/07_04_2022/fuzzy_svm/Processing_Data/dataset/yeast.csv')
    dataset_desc = dataset.describe(include = 'all')
    yeast_map = {'ME2':1,'CYT':-1,'ERL':-1,'EXC':-1,'ME1':-1,'ME3':-1,'MIT':-1,'NUC':-1,'POX':-1,'VAC':-1}
    dataset['name'] = dataset['name'].map(yeast_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    #Split data
    #X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42, stratify=y)
    #Scalling Data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    
    return X_train, y_train, X_test, y_test



 
