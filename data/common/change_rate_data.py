# import library
import numpy as np
import pandas as pd


# Change rate of positive class in data

def change_rate_data(X, y, new_rate):
    #pos_Y luu thu tu cua y(+)
    #neg_y luu so thu tu cua y(-)
    pos_y = np.where(y == 1)[0]
    # permutation
    pos_y = np.random.permutation(pos_y)
    neg_y = np.where( y == -1 )[0]
    rate0 = pos_y.shape[0]/(y.shape[0])
    pos_y_choosed =int(pos_y.shape[0]-(pos_y.shape[0] - new_rate * y.shape[0])/( 1 -new_rate) )
    pos_y =  pos_y[0:pos_y_choosed]
    #gop index 
    y_index = np.concatenate((neg_y, pos_y), axis = None)
    X = X[y_index]
    y = y[y_index]
    return X, y


# data = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/FUZZY SVM/CODE/07_04_2022/fuzzy_svm/data/datasets/Vertebral_column.csv')
# diag_map = {'Abnormal': -1.0, 'Normal': 1.0}
# data['Label class'] = data['Label class'].map(diag_map)
# X = data.values[:, 0:-1]
# y = data.values[:, 6]

# pos_y = np.where(y == 1)[0]
# print(pos_y)
# pos_y = np.random.permutation(pos_y)
# print(pos_y)
# print(pos_y.shape[0])
# print(y.shape[0])
# pos_y_choosed =int(pos_y.shape[0]-(pos_y.shape[0] - 1/5 * y.shape[0])/( 1 -1/5))
# print(pos_y_choosed)
# pos_y =  pos_y[0:pos_y_choosed]
# print(pos_y)