# import library
import numpy as np
import pandas as pd

# Change rate of positive class in data

def change_rate_data(X, y, new_rate = 1/ 15):
    #pos_Y luu thu tu cua y(+)
    #neg_y luu so thu tu cua y(-)
    pos_y = np.where(y == 1)[0]
    # permutation
    pos_y = np.random.permutation(pos_y)
    neg_y = np.where( y == -1 )[0]
    rate0 = pos_y.shape[0]/(y.shape[0])
    # pos_y_choosed =int((pos_y.shape[0] - new_rate * y.shape[0])/( 1 -new_rate) ) 
    pos_y_choosed = int(neg_y.shape[0]*new_rate)
    pos_y =  pos_y[0:pos_y_choosed]
    #gop index 
    y_index = np.concatenate((pos_y, neg_y), axis = None)
    X = X[y_index]
    y = y[y_index]
    return X, y
    
# dataset = pd.read_csv("D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/FUZZY SVM/CODE/07_04_2022/fuzzy_svm/Processing_Data/dataset/Co_Author_50_250.csv")
# X = dataset.values[:, 0:-1]
# y = dataset.values[:, 7]

# pos_y = np.where(y == 1)[0]
# pos_y = np.random.permutation(pos_y)
# print(pos_y)
# neg_y = np.where( y == -1 )[0]
# pos_y_choosed = int(neg_y.shape[0]*1/5)
# print(pos_y_choosed)
# pos_y =  pos_y[0:pos_y_choosed]
# # pos_y.sort()
# # print(pos_y)
# y_index = np.concatenate((neg_y, pos_y), axis = None)
# print(y_index)

# X = X[y_index]
# y = y[y_index]

# print(X)
# print(X[50])
# print(y)
# print(y[50])