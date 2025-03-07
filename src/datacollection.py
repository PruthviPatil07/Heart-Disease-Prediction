#Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report

df=pd.read_csv(r"C:\Users\PRUTHVIRAJ\Documents\Code Pruthvi\heart_disease_data.csv")

import os
os.makedirs(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\raw_data',exist_ok=True)

df.to_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\raw_data\raw.csv',index=False)
