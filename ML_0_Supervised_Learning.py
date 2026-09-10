"""
=== SUPERVİSED LEARNING - Denetimli Öğrenme (Features(+), Label(+)) ===
Bir öğrencinin:
1-) Günlük çalışma saati
2-) Derse katılım yüzdesi
bu bilgilere bakarak sınavı geçip geçmeyeceğini tahmin edelim.

Label:
    0 : Kaldı
    1 : Geçti

"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def main():

    # 1. parametre: Günlük Çalışma Saati
    # 2. parametre: Derse Katılım Yüzdesi
    X = np.array([
        [1, 30],
        [2, 40],
        [2, 50],
        [3, 55],
        [4, 60],
        [5, 65],
        [6, 75],
        [7, 85],
        [8, 90],
        [9, 95]
    ])

    # 0 -> Kaldı, 1 -> Geçti
    y = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    print("=== SUPERVISED LEARNING (Features(+), Label(+)) ===")
    print("\nX - Öğrenci Özellikler(Features)")
    print(X)

    print("\ny - Label(Etiketler)")
    print(y)


    model = LogisticRegression() # Boş YZ Beyni
    model.fit(X,y) # Model Eğitimi

    # Örnek: Öğrenci 6 saat çalışıyor, Derse katılım %80
    new_student = np.array([[6, 80]])
    prediction = model.predict(new_student)[0]
    probabilities = model.predict_proba(new_student)[0]

    print("\nYENİ ÖĞRENCİ")
    print("Çalışma Saati: 6 saat")
    print("Derse katılım: %80")

    print("\nModel Tahmini:", prediction)

    if prediction == 1:
        print("Öğrencinin sınavdan GEÇMESİ bekleniyor.")
    else:
        print("Öğrencinin sınavdan KALMASI bekleniyor.")

    print("\nOlasılıklar:")
    print(f"Kalma Olasılığı: %{probabilities[0] * 100:.2f}")
    print(f"Geçme Olasılığı: %{probabilities[1] * 100:.2f}")

if __name__== "__main__":
    main()
