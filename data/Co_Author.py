import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from ..common.change_rate_data import  change_rate_data
#from data.common.change_rate_data import change_rate_data
def load_data(test_size,testsize_val):
    data = pd.read_csv('./data/datasets/CoAuthor_100_500.csv')
    #print(data)
    diag_map = {-1: -1.0, 1: 1.0}
    data['Label class'] = data['Label class'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 7]
    #X, y = change_rate_data(X, y , new_rate = new_rate)
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    X_train_val, X_test_val, y_train_val, y_test_val = tts(X_train, y_train, test_size=testsize_val, random_state=42,stratify=y_train)
    # X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42)
    return X_train_val, y_train_val, X_test_val, y_test_val, X_test, y_test
# X_train,y_train, X_test, y_test,_,__ = load_data(test_size =0.5,testsize_val=0.2)
# print(X_train.shape)
# print(X_test.shape)