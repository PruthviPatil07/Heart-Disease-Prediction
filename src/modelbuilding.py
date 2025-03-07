import pandas as pd

X_train = pd.read_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\x_train.csv')
Y_train = pd.read_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\y_train.csv')
X_test = pd.read_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\x_test.csv')
Y_test = pd.read_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\y_test.csv')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
import joblib

models=[DecisionTreeClassifier(random_state=42),RandomForestClassifier(random_state=42),AdaBoostClassifier()]
d={}
for i in models:
    i.fit(X_train,Y_train)
    pred = i.predict(X_test)
    ac = accuracy_score(Y_test,pred)
    if i not in d:
        d[i] = ac
        
model = [a for a,b in d.items() if max(d.values()) == b][0]      
        
print(model)

import os 
os.makedirs(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\models', exist_ok=True)

joblib.dump(model,r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\models\model')