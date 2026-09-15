from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 1. Veriyi yükle (Features ve Labels)
iris = load_iris()
X = iris.data    # Çiçeğin boyutları (Features)
y = iris.target  # Çiçek türleri (Labels)

# 2. Modeli kur ve eğit (Training)
model = DecisionTreeClassifier()
model.fit(X, y)

# 3. Yeni verilerle tahmin yap (Inference)
yeni_cicek = [[5.2, 3.1, 4.2, 1.5]]
cikti = model.predict(yeni_cicek)
print("Tahmin Edilen Tür Kodu:", cikti)


