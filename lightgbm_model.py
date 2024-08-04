from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import time

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","Division","NewLeague"]])
y=datas["Salary"]
x_=datas.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=99)

lgbm=LGBMRegressor()
lgbm_params={
    "learning_rate":[0.01,0.1,0.5,1],
    "n_estimators":[20,40,50,60,70],
    "max_depth":[1,2,3,4,5,6]
}

lgbm_cv=GridSearchCV(lgbm,lgbm_params,cv=10,verbose=2,n_jobs=-1)
lgbm_cv.fit(x_train,y_train)
learning_rate=lgbm_cv.best_params_["learning_rate"]
n_estimators=lgbm_cv.best_params_["n_estimators"]
max_depth=lgbm_cv.best_params_["max_depth"]
lgbm_tuned=LGBMRegressor(max_depth=max_depth,n_estimators=n_estimators,learning_rate=learning_rate)
lgbm_tuned.fit(x_train,y_train)
predict=lgbm_tuned.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,predict))

print(rmse)





