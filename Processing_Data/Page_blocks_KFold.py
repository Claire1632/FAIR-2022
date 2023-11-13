import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from common.change_rate_data import change_rate_data

def load_data(test_size,testsize_val):
    dataset = pd.read_csv('./Processing_Data/dataset/page-blocks.csv')
    dataset_desc = dataset.describe(include = 'all')
    pageblocks_map = {5 : 1, 1: -1, 2: -1, 3: -1, 4: -1}
    dataset['class'] = dataset['class'].map( pageblocks_map)
    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    X = np.array(X)
    y = np.array(y)
    return X, y