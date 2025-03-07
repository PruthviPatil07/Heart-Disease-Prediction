import pandas as pd
from sklearn.model_selection import train_test_split 

df = pd.read_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\cleaned_data\cleaned.csv')

X=df.drop(columns='target',axis=1)
Y=df['target']

y = df['target']

# Split dataset into training set and test set
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2)

import os 

os.makedirs(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess', exist_ok=True)

X_train.to_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\x_train.csv',index=False)
Y_train.to_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\y_train.csv',index=False)
X_test.to_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\x_test.csv',index=False)
Y_test.to_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\y_test.csv',index=False)