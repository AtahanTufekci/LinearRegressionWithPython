import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *

def normalize(list_): # normalizasyon işlemi

    for i in range(len(list_)):
        list_[i] = (list_[i] - min(list_)) / (max(list_) - min(list_))

    return list_

def singleVarLinearReg1(t_0,t_1,x_1): #hipotez fonk. hesabı

    h_x = (t_0 * 1) + (t_1 * x_1)

    return h_x

def singleVarLinearReg2(t_0,t_1,x_1):
    h_x = (t_0 * 1) + (t_1 * np.sqrt(x_1))

    return h_x

def costFunc(t,matrix_): # bedel fonk.
    j_t = sum((t.dot(x) - y) ** 2 for x,y in matrix_)/(2 * len(matrix_))

    return j_t

def derCostFunc(t,matrix_): #bedel fonk. türevi
    j_t = sum((t.dot(x) - y) * x[1] for x,y in matrix_)/(len(matrix_))

    return j_t

def derCostFuncReg(t,matrix_): # bedel fonk. düzenlileştirilmesi
    j_t = sum((t.dot(x) - y) * x[1] + (10 / len(matrix_) * t[1]) for x,y in matrix_)/(len(matrix_))

    return j_t

def normalEquation(x,y): # normal denklem yaklaşımı
    x = np.array(x)
    y = np.array(y)
    t = ((((x.T).dot(x))**-1).dot(x.T)).dot(y)

    return t[0], t[1]

def gradientDescent(costFunc, derCostFunc, matrix_): # eğim azalması algoritması
    t = np.array([-0.1,0.53])
    alpha = 0.01
    value_list = []

    for i in range(180):
        value = costFunc(t,matrix_)
        gradient = derCostFunc(t,matrix_)
        t = t - alpha * gradient
        #print('Iteration {}: t = {}, F(t) = {} '.format(i, t, value))
        value_list.append(value)
    
    return t[0], t[1], value_list # öğrenilen parametreler
    
def regression_experiments(points): # verilerin  5-kat çapraz geçerleme ile bölünmesi
    train_list_x = []
    train_list_y = []
    test_list_x = []
    test_list_y = []

    points = np.asarray(points) 
    kf = KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(points):
        X_train, X_test = points[train_index,0], points[test_index,0]
        y_train, y_test = points[train_index,1], points[test_index,1]  
        train_list_x.append(X_train)
        train_list_y.append(y_train)
        test_list_x.append(X_test)
        test_list_y.append(y_test)

    return train_list_x, train_list_y, test_list_x, test_list_y      

def gradientDescentReg(costFunc, derCostFunc1, derCostFunc2, matrix_): # eğim azalması algoritması (düzenlileştirme ile)
    t = np.array([-0.1,0.53])
    alpha = 0.01
    value_list = []

    for i in range(180):
        value = costFunc(t,matrix_)
        gradient1 = derCostFunc1(t,matrix_)
        gradient2 = derCostFunc2(t,matrix_)
        t[0] = t[0] - alpha * gradient1
        t[1] = t[1] - alpha * gradient2
        #print('Iteration {}: t = {}, F(t) = {} '.format(i, t, value))
        value_list.append(value)
    
    return t[0], t[1], value_list # öğrenilen parametreler

def calculateMAE(t_0,t_1,list_,y,singleVarLinearReg): # mean absolute error hesabının yapılması

    regression_list = []

    for i in range(len(list_)):
        regression_list.append(singleVarLinearReg(t_0,t_1,list_[i])) # regresyon değerlerini hesapla   

    return mean_absolute_error(y,regression_list)

def returnReg(t_0,t_1,list_,singleVarLinearReg): # regresyon değerlerini döner

    regression_list = []

    for i in range(len(list_)):
        regression_list.append(singleVarLinearReg(t_0,t_1,list_[i])) # regresyon değerlerini hesapla  

    return regression_list    
