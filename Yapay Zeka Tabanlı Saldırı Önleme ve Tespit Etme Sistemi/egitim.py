import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib

# Veri setinin yüklenmesi
data = pd.read_excel("/home/kali/Desktop/22040301116/agtrafigi.xlsx") # Tam dosya yolunu kullanın

# Bağımsız ve bağımlı değişkenlerin ayrılması 
features = data.drop(columns="Label")  # Bağımsız değişkenler
target = data["Label"]  # Hedef değişken

# Eğitim ve test setlerinin oluşturulması
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42)

# IP adreslerini sayısal değerlere dönüştürme işlemi
def ip_to_numeric(ip):
    return sum([int(part) * (256 ** idx) for idx, part in enumerate(ip.split('.')[::-1])])

X_train_ip = X_train['Source'].apply(ip_to_numeric).values.reshape(-1, 1)
X_test_ip = X_test['Source'].apply(ip_to_numeric).values.reshape(-1, 1)

# Ölçeklendirme
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_ip)
X_test_scaled = scaler.transform(X_test_ip)

# Karar ağacı sınıflandırıcısı
decision_tree = DecisionTreeClassifier()

# Modeli eğitme
decision_tree.fit(X_train_scaled, y_train)

# Tahmin yapma
predictions = decision_tree.predict(X_test_scaled)

# Doğruluk skorunu hesapla
accuracy = accuracy_score(y_test, predictions)
print("Model Doğruluğu:", accuracy)

# Modeli kaydet
joblib.dump(decision_tree, "trained_decision_tree_model.joblib")

# Modeli yükle
loaded_model = joblib.load("trained_decision_tree_model.joblib")

# Yüklenen modelle tahmin yap
loaded_predictions = loaded_model.predict(X_test_scaled)

# Yüklenen modelin doğruluk skorunu hesapla
loaded_accuracy = accuracy_score(y_test, loaded_predictions)
print("Yüklenen Model Doğruluğu:", loaded_accuracy)

