# -*- coding: utf-8 -*-
import pandas as pd
import joblib
import sys
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Modeli yükle
model = joblib.load(r"C:/Users/FD Bilgisayar/Desktop/proje/rgr_tree_model.pkl")

# Komut satırından parametreler: store, item, year, month, day
if len(sys.argv) > 5:
    store = int(sys.argv[1])
    item = int(sys.argv[2])
    year = int(sys.argv[3])
    month = int(sys.argv[4])
    day = int(sys.argv[5])
else:
    # Test için default değerler
    store = 1
    item = 25
    year = 2025
    month = 8
    day = 20

# Tahmin için dataframe
df_input = pd.DataFrame([[store, item, year, month, day]], columns=['store','item','year','month','day'])

# Tahmin
prediction = model.predict(df_input)
print(int(prediction[0]))

