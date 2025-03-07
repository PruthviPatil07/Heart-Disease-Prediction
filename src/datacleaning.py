import pandas as pd

df = pd.read_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\raw_data\raw.csv')

import os 

os.makedirs(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\cleaned_data',exist_ok=True)

df.to_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\cleaned_data\cleaned.csv', index=False)

