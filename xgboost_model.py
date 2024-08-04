import xgboost 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import numpy as np
datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","Division","NewLeague"]])
y=datas["Salary"]
x_=datas.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

xgbr=XGBRegressor()
xgb_params={
    "learning_rate":[0.1,0.01,0.5],#overfitting i engeller öğrenme oranı daraltma adım boyunu ifade eder
    "max_depth":[2,3,4,5],#max derinilk
    "n_estimators":[100,200,500,1000],#kullanılacak ağac sayısı
    "colsample_bytree":[0.4,0.7,1]#oluşturulacak ağaclardan alınacak alt küme oranını verir
}
#en iyi parametreleri alacağız
gxb_cv=GridSearchCV(xgbr,xgb_params,cv=5,n_jobs=-1,verbose=2)
gxb_cv.fit(x_train,y_train)
learning_rate=gxb_cv.best_params_["learning_rate"]
max_depth=gxb_cv.best_params_["max_depth"]
n_estimators=gxb_cv.best_params_["n_estimators"]
colsample_bytree=gxb_cv.best_params_["colsample_bytree"]
#tuned model
xgb_tuned=XGBRegressor(colsample_bytree=colsample_bytree,n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
xgb_tuned.fit(x_train,y_train)
predict=xgb_tuned.predict(x_test)
RMSE=np.sqrt(mean_squared_error(y_test,predict))
print(RMSE)






