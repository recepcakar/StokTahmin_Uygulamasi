# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:02:55 2025

@author: Recep Çakar
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import pandas as pd
import sys
from datetime import datetime, timedelta

# Modeli yükle (sadece bir kez)
model = joblib.load(r"C:/Users/FD Bilgisayar/Desktop/proje/rgr_tree_model.pkl")

# Komut satırından parametreler: store, item, start_year, start_month, start_day, max_days
if len(sys.argv) > 6:
    store = int(sys.argv[1])
    item = int(sys.argv[2])
    year = int(sys.argv[3])
    month = int(sys.argv[4])
    day = int(sys.argv[5])
    max_days = int(sys.argv[6])
else:
    store, item, year, month, day = 1, 25, 2025, 8, 20
    max_days = 100  # Tahmin için maksimum gün sayısı

start_date = datetime(year, month, day)

# Günlük tahminleri üst üste ekleyerek stok bitiş gününü bul
stok_miktari = int(sys.argv[7])  # C# tarafından gönderilecek stok miktarı
toplam = 0
gun_sayisi = 0
tarih = start_date

while toplam < stok_miktari and gun_sayisi < max_days:
    df_input = pd.DataFrame([[store, item, tarih.year, tarih.month, tarih.day]],
                            columns=['store','item','year','month','day'])
    tahmin = int(model.predict(df_input)[0])
    toplam += tahmin
    gun_sayisi += 1
    tarih += timedelta(days=1)

# Sonuç olarak stok bitiş tarihini ve geçen gün sayısını yazdır
print(f"{tarih.strftime('%Y-%m-%d')},{gun_sayisi}")
