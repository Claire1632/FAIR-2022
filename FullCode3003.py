import numpy as np
from data import Vertebral_column
from data import Co_Author
from Processing_Data import Abanole
from Processing_Data import Ecoli
from Processing_Data import Ecloli1
from Processing_Data import Ecoli3
from Processing_Data import Glass1
from Processing_Data import Glass4
from Processing_Data import Haberman
from Processing_Data import Waveform
from Processing_Data import New_thyroid2
from Processing_Data import Page_blocks
from Processing_Data import Pima_Indians_Diabetes
from Processing_Data import Satimage
from Processing_Data import Transfusion
from Processing_Data import Yeast
from Processing_Data import Transfution_Kfold
from Processing_Data import Satimage_KFold
from sklearn.preprocessing import StandardScaler
from Processing_Data import Ecoli_Kfold
from Processing_Data import Pima
from Processing_Data import Haberman_KFold
from Processing_Data import Yeast_KFold
from data import indian_liver_patient
#from data import spect_heart
from wsvm.application import Wsvm
from svm.application import Svm
from sklearn.svm import SVC
#from sklearn.metrics import f1_score
from collections import Counter
from sklearn.metrics  import classification_report,precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score,f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from sklearn import metrics
import math
from datetime import datetime
from fuzzy.weight import fuzzy
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedShuffleSplit

def svm_lib(X_train, y_train,X_test):
    svc=SVC(probability=True, kernel='linear')
    model = svc.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def wsvm(C,X_train, y_train,X_test,distribution_weight=None):
    model = Wsvm(C,distribution_weight)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    return test_pred

def svm(C,X_train, y_train,X_test):
    model = Svm(C)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    return test_pred

def Gmean(y_test,y_pred):
    cm_WSVM = metrics.confusion_matrix(y_test, y_pred)
    sensitivity = cm_WSVM[1,1]/(cm_WSVM[1,0]+cm_WSVM[1,1])
    specificity = cm_WSVM[0,0]/(cm_WSVM[0,0]+cm_WSVM[0,1])
    gmean = math.sqrt(sensitivity*specificity)
    return specificity,sensitivity,gmean

def metr(X_train,y_test,test_pred,sp,se,gmean):
    #se,sp,gmean = Gmean(y_test,test_pred)
    print("So luong samples: ",len(X_train))
    print("\n",classification_report(y_test, test_pred))
    print("SP      : ",sp)
    print("SE      : ",se)
    print("Gmean   : ",gmean)
    print("F1 Score: ",f1_score(y_test, test_pred))
    print("Accuracy: ",accuracy_score(y_test,test_pred))
    print("AUC     : ",roc_auc_score(y_test, test_pred))
    print("Ma tran nham lan: \n",confusion_matrix(y_test, test_pred))

def metr_text(f,X_train,y_test,test_pred,sp,se,gmean):
    #se,sp,gmean = Gmean(y_test,test_pred)
    f.write(f"\n\nSo luong samples Tong: {len(X_train)+len(y_test)}")
    f.write(f"\n\nSo luong samples training: {len(X_train)}")
    f.write(f"\nSo luong samples testing: {len(y_test)}\n")
    # f.write(f"\nSo luong positive samples training: {Counter(y.where(y_train==1))}\n")
    # f.write(f"\nSo luong negative samples training: {Counter(y.where(y_train == -1))}\n")
    # f.write(f"\nSo luong positive samples testing: {Counter(y.where(y_test==1))}\n")
    # f.write(f"\nSo luong negative samples testing: {Counter(y.where(y_test == -1))}\n")
    f.write("\n"+str(classification_report(y_test, test_pred)))
    f.write(f"\nSP      : {sp:0.4f}")
    f.write(f"\nSE      : {se:0.4f}")
    f.write(f"\nGmean   : {gmean:0.4f}")
    f.write(f"\nF1 Score: {f1_score(y_test, test_pred):0.4f}")
    f.write(f"\nAccuracy: {accuracy_score(y_test,test_pred):0.4f}")
    f.write(f"\nAUC     : {roc_auc_score(y_test, test_pred):0.4f}")
    f.write("\n\nMa tran nham lan: \n"+str(confusion_matrix(y_test, test_pred)))


def compute_weight(X, y, name_method, name_function, beta=None, C=None, gamma=None, u=None, sigma=None):
    method = fuzzy.method()
    function = fuzzy.function()
    pos_index = np.where(y == 1)[0]
    neg_index = np.where(y == -1)[0]
    try:
        if name_method == "own_class_center":
            d = method.own_class_center(X, y)
        elif name_method == "estimated_hyper_lin":  # actual_hyper_lin, own_class_center
            d = method.estimated_hyper_lin(X, y)
        elif name_method == 'actual_hyper_lin':
            d = method.actual_hyper_lin(X, y, C=C, gamma=gamma)
        elif name_method == "distance_center_own_opposite_tam":
            d_own, d_opp, d_tam = method.distance_center_own_opposite_tam(X, y)
        else:
            print('dont exist method')

        if name_function == "lin":
            W = function.lin(d)
        elif name_function == "exp":
            W = function.exp(d, beta)
        elif name_function == "lin_center_own":
            W = function.lin_center_own(d, pos_index, neg_index)
        elif name_function == 'gau':
            W = function.gau(d, u, sigma)
        elif name_function == "func_own_opp":
            W = function.func_own_opp(d_own, d_opp, pos_index, neg_index, d_tam)
        # elif name_function == "func_own_opp_new":
        #     W = function.func_own_opp_new(d_own, d_opp, pos_index, neg_index, d_tam)
    except Exception as e:
        print('dont exist function')
        print(e)
    pos_index = np.where(y == 1)[0]
    neg_index = np.where(y == -1)[0]
    r_pos = 1
    r_neg = len(pos_index) / len(neg_index)
    m = []
    W = np.array(W)
    m = W[pos_index] * r_pos
    m = np.append(m, W[neg_index] * r_neg)
    return m

def own_opp(X,y):
    pos_index = np.where(y == 1)[0]
    neg_index = np.where(y == -1)[0]
    method = fuzzy.method()
    function = fuzzy.function()
    d_own, d_opp, d_tam = method.distance_center_own_opposite_tam(X,y)
    # W = function.func_own_opp_new(d_own,d_opp,pos_index,neg_index,d_tam)
    W = function.func_own_opp(d_own, d_opp, pos_index, neg_index, d_tam)
    return W

def fuzzy_weight(f,beta_center, beta_estimate, beta_actual,X_train, y_train,namemethod,namefunction):
    if namemethod =="own_class_center" and namefunction == "exp":
        distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction,beta = beta_estimate)
        f.write(f"\n\t Beta 'own_class_center' with exp = {beta_center}\n")
    elif namemethod =="estimated_hyper_lin" and namefunction == "exp":
        distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction,beta = beta_estimate)
        f.write(f"\n\t Beta 'estimated_hyper_lin' with exp = {beta_estimate}\n")
    elif namemethod =="actual_hyper_lin" and namefunction == "exp":
        distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction,beta = beta_actual)
        f.write(f"\n\t Beta 'actual_hyper_lin' with exp = {beta_actual}\n")
    else:
        distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction)
    return distribution_weight


C = 100
# thamso1 = 1
# thamso2 = 1
# T = 3
# n_neighbors = 5
# test_size = [0.2, 0.3, 0.4]
# testsize_val = 0.2
# data = [Co_Author, Abanole, Ecoli, Ecloli1, Ecoli3, Glass1, Glass4, Haberman, Waveform, New_thyroid2, Page_blocks,
# #             Pima_Indians_Diabetes, Satimage, Transfusion, Yeast]
# data = [Haberman]
# data = [Pima_Indians_Diabetes]
# data = [Transfusion]
# data = [Ecoli]
# data = [Abanole]
# data = [Yeast]
# data = [Pima]
data = [Ecoli_Kfold]
# !!!!!!! Beta with Dataset, change Data please change Beta !!!!!!!!
beta_center, beta_estimate, beta_actual = 0.3, 0.6, 0.7 # !!!!!!! Beta with Dataset, change Data please change Beta !!!!!!!!
# !!!!!!! Beta with Dataset, change Data please change Beta !!!!!!!

# name_method = ["own_class_center", "estimated_hyper_lin", "actual_hyper_lin", "distance_center_own_opposite_tam"]
# name_function = ["lin_center_own", "exp", "func_own_opp_new"]
# name_method =["own_class_center_divided"]
# name_method = ["distance_center_own_opposite_tam"]
# name_function = ["func_own_opp_new"]
name_method = ["own_class_center", "estimated_hyper_lin", "actual_hyper_lin", "distance_center_own_opposite_tam"]
name_function = ["lin","lin_center_own", "exp", "func_own_opp"]

time = datetime.now().strftime("%d%m%Y_%H%M%S")
filepath = "./text_script"

# svc lib
svc = SVC(probability=True, kernel='linear')
# svm scratch
svm_scr = Svm(C)
# W.svm

for dataset in data:
    filename = (str(dataset).split("\\")[-1]).split(".")[0]
    f = open(f"{filepath}/Data_{filename}_{time}_Detail.txt", "w")
    # f.write(f"\nC = {C}, thamso1 = {thamso1}, thamso2 = {thamso2}, T = {T}, n_neighbors = {n_neighbors}  \n")
    f.write(f"\n\n\tUSING DATASET : {filename}\n")

    f2 = open(f"{filepath}/Data_{filename}_{time}_Main.txt", "w")
    # f2.write(f"\nC = {C}, thamso1 = {thamso1}, thamso2 = {thamso2}, T = {T}, n_neighbors = {n_neighbors}  \n")
    f2.write(f"\n\n\tUSING DATASET : {filename}\n")

    # X_train, X_test,y_train, y_test = dataset.load_data()

    X, y = dataset.load_data()
    # kfold_validation = KFold(n_splits=5, shuffle=True)
    # kfold_validation = StratifiedKFold(n_splits=5, shuffle=True) #3
    # kfold_validation = StratifiedKFold(n_splits=5) #2
    kfold_validation = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0) #1
    # kfold_validation = [kfold_validation1,kfold_validation2,kfold_validation3]
    for train_index, test_index in kfold_validation.split(X,y):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        # Scalling Data
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        # Svm library
        f.write("\n\n\tSVM LIBRARY starting...\n")
        f2.write("\n\n\tSVM LIBRARY starting...\n")
        print("SVM LIBRARY starting...\n")
        test_pred = svm_lib(X_train, y_train, X_test)
        sp, se, gmean = Gmean(y_test, test_pred)
        # metr(X_train_val,y_test,test_pred,sp,se,gmean)
        metr_text(f, X_train, y_test, test_pred, sp, se, gmean)
        metr_text(f2, X_train, y_test, test_pred, sp, se, gmean)

        # Svm scratch
        # f.write("\n\n\tSVM starting...\n")
        # f2.write("\n\n\tSVM starting...\n")
        # print("SVM starting...\n")
        # test_pred = svm(C, X_train, y_train, X_test)
        # sp, se, gmean = Gmean(y_test, test_pred)
        # # metr(X_train_val,y_test,test_pred,sp,se,gmean)
        # metr_text(f, X_train, y_test, test_pred, sp, se, gmean)
        # metr_text(f2, X_train, y_test, test_pred, sp, se, gmean)

        # Wsvm
        f.write("\n\n\tW.SVM starting...\n")
        f2.write("\n\n\tW.SVM starting...\n")
        print("W.SVM starting...\n")
        N, d = X_train.shape
        distribution_weight = np.ones(N)
        test_pred = wsvm(C, X_train, y_train, X_test, distribution_weight)
        sp, se, gmean = Gmean(y_test, test_pred)
        # metr(X_train_val,y_test,test_pred,sp,se,gmean)
        metr_text(f, X_train, y_test, test_pred, sp, se, gmean)
        metr_text(f2, X_train, y_test, test_pred, sp, se, gmean)

        # FuzyyWsvm
        for namemethod in name_method:
            for namefunction in name_function:
                if namemethod == "distance_center_own_opposite_tam" and namefunction == "func_own_opp":
                    f.write(
                        f"\n\n\tFuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
                    f2.write(
                        f"\n\n\tFuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
                    print(f"Fuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
                    distribution_weight2 = own_opp(X_train, y_train)
                    test_pred = wsvm(C, X_train, y_train, X_test, distribution_weight2)
                    sp2, se2, gmean2 = Gmean(y_test, test_pred)
                    metr_text(f, X_train, y_test, test_pred, sp2, se2, gmean2)
                    metr_text(f2, X_train, y_test, test_pred, sp2, se2, gmean2)
                elif namemethod == "distance_center_own_opposite_tam" and namefunction == "lin_center_own":
                    continue
                elif namemethod == "distance_center_own_opposite_tam" and namefunction == "exp":
                    continue
                elif namemethod == "own_class_center" and namefunction == "func_own_opp":
                    continue
                elif namemethod == "estimated_hyper_lin" and namefunction == "func_own_opp":
                    continue
                elif namemethod == "actual_hyper_lin" and namefunction == "func_own_opp":
                    continue
                elif namemethod == "own_class_center" and namefunction == "lin":
                    continue
                elif namemethod == "estimated_hyper_lin" and namefunction == "lin_center_own":
                    continue
                elif namemethod == "actual_hyper_lin" and namefunction == "lin_center_own":
                    continue
                elif namemethod == "distance_center_own_opposite_tam" and namefunction == "lin":
                    continue
                else:
                    f.write(
                        f"\n\n\tFuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
                    f2.write(
                        f"\n\n\tFuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
                    print(f"Fuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
                    distribution_weight = fuzzy_weight(f, beta_center, beta_estimate, beta_actual, X_train, y_train, namemethod, namefunction)
                    __ = fuzzy_weight(f2, beta_center, beta_estimate, beta_actual, X_train, y_train, namemethod, namefunction)
                    test_pred = wsvm(C, X_train, y_train, X_test, distribution_weight)
                    sp, se, gmean = Gmean(y_test, test_pred)
                    # metr(X_train_val,y_test,test_pred,sp,se,gmean)
                    metr_text(f, X_train, y_test, test_pred, sp, se, gmean)
                    metr_text(f2, X_train, y_test, test_pred, sp, se, gmean)
