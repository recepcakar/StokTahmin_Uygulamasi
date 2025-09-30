import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#  Veri Yükleme 
df = pd.read_csv(r"train dosyamın yolu")  # Türkçe klasör adı yerine güvenli yol kullandım

#  Tarih Dönüşümü 
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Eksik tarihleri atıyırum
df = df.dropna(subset=['date'])

# Tarihi parçala
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day


#  Özellikler ve Hedef 
X = df[['store', 'item', 'year', 'month', 'day',]]
y = df['sales']

# Eğitim / Test Ayrımı 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_model.fit(X_train, y_train)


# === Performans Testi ===
y_pred = tree_model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


joblib.dump(tree_model, r"C:kaydediceğimiz dizin")
print("Model kaydedildi!")

