import pandas as pd
from functions import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
from math import *


data1 = pd.read_csv("C:/Users/user/Desktop/odev2/machine.data")
x_1 = data1.NUMBER_OF_MACHINES
y = data1.PROFIT_LOSS_PER_MONTH

normalize(x_1) # özellikleri normalleşti
normalize(y) # hedefleri normalleştir

matrix1 = []

for i in range(x_1.shape[0]):
    matrix1.append((x_1[i],y[i]))


train_list_x, train_list_y, test_list_x, test_list_y = regression_experiments(matrix1) # 5-kat çapraz geçerleme ile verileri ayır

# X matrisi
n_matrix_x_1 = []
n_matrix_x_2 = []
n_matrix_x_3 = []
n_matrix_x_4 = []
n_matrix_x_5 = []

for i in range(len(train_list_x[0])):
    n_matrix_x_1.append((1,train_list_x[0][i])) # X matrisine ekleme
for i in range(len(train_list_x[1])):
    n_matrix_x_2.append((1,train_list_x[1][i])) # X matrisine ekleme
for i in range(len(train_list_x[2])):
    n_matrix_x_3.append((1,train_list_x[2][i])) # X matrisine ekleme
for i in range(len(train_list_x[3])):
    n_matrix_x_4.append((1,train_list_x[3][i])) # X matrisine ekleme
for i in range(len(train_list_x[4])):
    n_matrix_x_5.append((1,train_list_x[4][i])) # X matrisine ekleme

n_t0_1, n_t1_1 = normalEquation(n_matrix_x_1,train_list_y[0]) # 1. veri setinin train_x ve train_y değerleri ile normal denklem yaklaşımı
n_t0_2, n_t1_2 = normalEquation(n_matrix_x_2,train_list_y[1]) # 2. veri setinin train_x ve train_y değerleri ile normal denklem yaklaşımı
n_t0_3, n_t1_3 = normalEquation(n_matrix_x_3,train_list_y[2]) # 3. veri setinin train_x ve train_y değerleri ile normal denklem yaklaşımı
n_t0_4, n_t1_4 = normalEquation(n_matrix_x_4,train_list_y[3]) # 4. veri setinin train_x ve train_y değerleri ile normal denklem yaklaşımı
n_t0_5, n_t1_5 = normalEquation(n_matrix_x_5,train_list_y[4]) # 5. veri setinin train_x ve train_y değerleri ile normal denklem yaklaşımı 
     

# normalleştirilmiş ve ayrılmış x ve y train verilerini tutar
train_matrix_1 = []
train_matrix_2 = []
train_matrix_3 = []
train_matrix_4 = []
train_matrix_5 = []

for i in range(len(train_list_x[0])):
    train_matrix_1.append(((1,train_list_x[0][i]),train_list_y[0][i])) # bias terimi, x1 ve y değerleri       
for i in range(len(train_list_x[1])):
    train_matrix_2.append(((1,train_list_x[1][i]),train_list_y[1][i])) # bias terimi, x1 ve y değerleri  
for i in range(len(train_list_x[2])):
    train_matrix_3.append(((1,train_list_x[2][i]),train_list_y[2][i])) # bias terimi, x1 ve y değerleri  
for i in range(len(train_list_x[3])):
    train_matrix_4.append(((1,train_list_x[3][i]),train_list_y[3][i])) # bias terimi, x1 ve y değerleri  
for i in range(len(train_list_x[4])):
    train_matrix_5.append(((1,train_list_x[4][i]),train_list_y[4][i])) # bias terimi, x1 ve y değerleri  


t0_1, t1_1, value_list_1 = gradientDescent(costFunc, derCostFunc, train_matrix_1) # 1. veri setinin train_x ve train_y değerleri ile eğim azalması
t0_2, t1_2, value_list_2 = gradientDescent(costFunc, derCostFunc, train_matrix_2) # 2. veri setini train_x ve train_y değerleri ile eğim azalması
t0_3, t1_3, value_list_3 = gradientDescent(costFunc, derCostFunc, train_matrix_3) # 3. veri setini train_x ve train_y değerleri ile eğim azalması
t0_4, t1_4, value_list_4 = gradientDescent(costFunc, derCostFunc, train_matrix_4) # 4. veri setini train_x ve train_y değerleri ile eğim azalması
t0_5, t1_5, value_list_5 = gradientDescent(costFunc, derCostFunc, train_matrix_5) # 5. veri setini train_x ve train_y değerleri ile eğim azalması

# orjinal değerleri geri al
data1 = pd.read_csv("C:/Users/user/Desktop/odev2/machine.data")
x_2 = data1.NUMBER_OF_MACHINES
y_2 = data1.PROFIT_LOSS_PER_MONTH

matrix2 = []

for i in range(x_2.shape[0]):
    matrix2.append((x_2[i],y_2[i]))

train_list_x_2, train_list_y_2, test_list_x_2, test_list_y_2 = regression_experiments(matrix2)

mae_train = 0

mae_train += calculateMAE(t0_1,t1_1,train_list_x_2[0],train_list_y_2[0],singleVarLinearReg1)
mae_train += calculateMAE(t0_2,t1_2,train_list_x_2[1],train_list_y_2[1],singleVarLinearReg1)
mae_train += calculateMAE(t0_3,t1_3,train_list_x_2[2],train_list_y_2[2],singleVarLinearReg1) # eğitim veri setlerinin normal eğim azalması için
mae_train += calculateMAE(t0_4,t1_4,train_list_x_2[3],train_list_y_2[3],singleVarLinearReg1) # ortalama MAE değerlerinin hesaplanması
mae_train += calculateMAE(t0_5,t1_5,train_list_x_2[4],train_list_y_2[4],singleVarLinearReg1)
print(f"5 eğitim veri setinin NORMAL EĞİM AZALMASI için ortalama MAE(mean absolute error) değeri: {mae_train/5:.2f}")

mae_test = 0

mae_test += calculateMAE(t0_1,t1_1,test_list_x_2[0],test_list_y_2[0],singleVarLinearReg1)
mae_test += calculateMAE(t0_2,t1_2,test_list_x_2[1],test_list_y_2[1],singleVarLinearReg1)
mae_test += calculateMAE(t0_3,t1_3,test_list_x_2[2],test_list_y_2[2],singleVarLinearReg1) # test veri setlerinin normal eğim azalması için
mae_test += calculateMAE(t0_4,t1_4,test_list_x_2[3],test_list_y_2[3],singleVarLinearReg1) # ortalama MAE değerlerinin hesaplanması
mae_test += calculateMAE(t0_5,t1_5,test_list_x_2[4],test_list_y_2[4],singleVarLinearReg1)
print(f"5 test veri setinin NORMAL EĞİM AZALMASI için ortalama MAE(mean absolute error) değeri: {mae_test/5:.2f}")

r_t0_1, r_t1_1, r_value_list_1 = gradientDescentReg(costFunc, derCostFunc, derCostFuncReg, train_matrix_1) # 1. verilerin eğim azalması(düzenlileştirme ile)
r_t0_2, r_t1_2, r_value_list_2 = gradientDescentReg(costFunc, derCostFunc, derCostFuncReg, train_matrix_2) # 2. verilerin eğim azalması(düzenlileştirme ile)
r_t0_3, r_t1_3, r_value_list_3 = gradientDescentReg(costFunc, derCostFunc, derCostFuncReg, train_matrix_3) # 3. verilerin eğim azalması(düzenlileştirme ile)
r_t0_4, r_t1_4, r_value_list_4 = gradientDescentReg(costFunc, derCostFunc, derCostFuncReg, train_matrix_4) # 4. verilerin eğim azalması(düzenlileştirme ile)
r_t0_5, r_t1_5, r_value_list_5 = gradientDescentReg(costFunc, derCostFunc, derCostFuncReg, train_matrix_5) # 5. verilerin eğim azalması(düzenlileştirme ile)

mae_train = 0

mae_train += calculateMAE(r_t0_1,r_t1_1,train_list_x_2[0],train_list_y_2[0],singleVarLinearReg1)
mae_train += calculateMAE(r_t0_2,r_t1_2,train_list_x_2[1],train_list_y_2[1],singleVarLinearReg1)
mae_train += calculateMAE(r_t0_3,r_t1_3,train_list_x_2[2],train_list_y_2[2],singleVarLinearReg1) # eğitim veri setlerinin düzenlileştirilmiş eğim azalması
mae_train += calculateMAE(r_t0_4,r_t1_4,train_list_x_2[3],train_list_y_2[3],singleVarLinearReg1) # için ortalama MAE değerlerinin hesaplanması
mae_train += calculateMAE(r_t0_5,r_t1_5,train_list_x_2[4],train_list_y_2[4],singleVarLinearReg1)
print(f"5 eğitim veri setinin DÜZENLİLEŞTİRİLMİŞ EĞİM AZALMASI için ortalama MAE(mean absolute error) değeri: {mae_train/5:.2f}")

mae_test = 0

mae_test += calculateMAE(r_t0_1,r_t1_1,test_list_x_2[0],test_list_y_2[0],singleVarLinearReg1)
mae_test += calculateMAE(r_t0_2,r_t1_2,test_list_x_2[1],test_list_y_2[1],singleVarLinearReg1)
mae_test += calculateMAE(r_t0_3,r_t1_3,test_list_x_2[2],test_list_y_2[2],singleVarLinearReg1) # test veri setlerinin düzenlileştirilmiş eğim azalması
mae_test += calculateMAE(r_t0_4,r_t1_4,test_list_x_2[3],test_list_y_2[3],singleVarLinearReg1) # için ortalama MAE değerlerinin hesaplanması
mae_test += calculateMAE(r_t0_5,r_t1_5,test_list_x_2[4],test_list_y_2[4],singleVarLinearReg1)
print(f"5 test veri setinin DÜZENLİLEŞTİRİLMİŞ EĞİM AZALMASI için ortalama MAE(mean absolute error) değeri: {mae_test/5:.2f}")

mae_train = 0

mae_train += calculateMAE(n_t0_1,n_t1_1,train_list_x_2[0],train_list_y_2[0],singleVarLinearReg2)
mae_train += calculateMAE(n_t0_2,n_t1_2,train_list_x_2[1],train_list_y_2[1],singleVarLinearReg2)
mae_train += calculateMAE(n_t0_3,n_t1_3,train_list_x_2[2],train_list_y_2[2],singleVarLinearReg2) # eğitim veri setlerinin normal denklem yaklaşımı için
mae_train += calculateMAE(n_t0_4,n_t1_4,train_list_x_2[3],train_list_y_2[3],singleVarLinearReg2) # ortalama MAE değerlerinin hesaplanması
mae_train += calculateMAE(n_t0_5,n_t1_5,train_list_x_2[4],train_list_y_2[4],singleVarLinearReg2)
print(f"5 eğitim veri setinin NORMAL DENKLEM YAKLAŞIMI için ortalama MAE(mean absolute error) değeri: {mae_train/5:.2f}")

mae_test = 0

mae_test += calculateMAE(n_t0_1,n_t1_1,test_list_x_2[0],test_list_y_2[0],singleVarLinearReg2)
mae_test += calculateMAE(n_t0_2,n_t1_2,test_list_x_2[1],test_list_y_2[1],singleVarLinearReg2)
mae_test += calculateMAE(n_t0_3,n_t1_3,test_list_x_2[2],test_list_y_2[2],singleVarLinearReg2) # test veri setlerinin normal denklem yaklaşımı için
mae_test += calculateMAE(n_t0_4,n_t1_4,test_list_x_2[3],test_list_y_2[3],singleVarLinearReg2) # ortalama MAE değerlerinin hesaplanması
mae_test += calculateMAE(n_t0_5,n_t1_5,test_list_x_2[4],test_list_y_2[4],singleVarLinearReg2)
print(f"5 test veri setinin NORMAL DENKLEM YAKLAŞIMI için ortalama MAE(mean absolute error) değeri: {mae_test/5:.2f}")

plt.scatter(train_list_x_2[0],train_list_y_2[0])
plt.plot(train_list_x_2[0],returnReg(t0_1,t1_1,train_list_x_2[0],singleVarLinearReg1),color="red")
plt.xlabel("X1 Özelliği")
plt.ylabel("1. Eğitim Veri Seti")
plt.title("Regresyon Grafiği(Normal Eğim Azalması)")
plt.show()

plt.scatter(train_list_x_2[0],train_list_y_2[0])
plt.plot(train_list_x_2[0],returnReg(r_t0_1,r_t1_1,train_list_x_2[0],singleVarLinearReg1),color="red")
plt.xlabel("X1 Özelliği")
plt.ylabel("1. Eğitim Veri Seti")
plt.title("Regresyon Grafiği(Düzenlileştirilmiş Eğim Azalması)")
plt.show()

plt.scatter(train_list_x_2[0],train_list_y_2[0])
plt.plot(train_list_x_2[0],returnReg(n_t0_1,n_t1_1,train_list_x_2[0],singleVarLinearReg2),color="red")
plt.xlabel("X1 Özelliği")
plt.ylabel("1. Eğitim Veri Seti")
plt.title("Regresyon Grafiği(Normal Denklem Yaklaşımı)")
plt.show()

liste = [i for i in range(1,181)]

plt.plot(liste,value_list_1)
plt.xlabel("Döngü Sayısı")
plt.ylabel("Bedel Fonskiyonu")
plt.title("Bedel Fonksiyonu Grafiği(Normal Eğim Azalması)")
plt.show()

plt.plot(liste,r_value_list_1)
plt.xlabel("Döngü Sayısı")
plt.ylabel("Bedel Fonskiyonu")
plt.title("Bedel Fonksiyonu Grafiği(Düzenlileştirilmiş Eğim Azalması)")
plt.show()