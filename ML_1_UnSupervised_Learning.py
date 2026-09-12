"""
=== UNSUPERVİSED LEARNING - Denetimsiz Öğrenme (Features(+), Label(-)) ===

Müşterilerin:
1-) Yıllık Gelir
2-) Aylık Harcama
bu bilgilere bakarak müşterileri 3 gruba ayıracağız.

Ancak modele:
Bu müşteri A grubundadır, bu müşteri B grubundadır gibi hiç bir doğru cevap vermesin.

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():

    # 1. değer: Yıllık Gelir
    # 2. değer: Aylık Harcama
    X = np.array([
        [20, 10],
        [22, 12],
        [25, 15],

        [50, 45],
        [52, 48],
        [55, 50],

        [85, 80],
        [88, 85],
        [90, 88],
    ])

    print("=== UNSUPERVISED LEARNING ===")
    print("\nMüşteri Verileri:")
    print(X)

# --------------------------------------------
    # ÖLÇEKLEME
    # StandardScaler, farklı büyüklükteki sayıları benzer ölçeğe getirir.
    # K-Means uzaklık hesabı yaptığı için ölçekleme faydalıdır.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# --------------------------------------------
    # K-Means Modeli
    # 3  = Kümeleme
    # 42 = Rastgele yapılan işlemlerin her çalışmada aynı sonucu vermesini sağlar.
    # 10 = 10 farklı başlangıcı dene ve en iyisini seç
    model = KMeans(
        n_clusters = 3,
        random_state = 42,
        n_init = 10
    )

    # Model hem öğrenir, hem de veri için bir cluster numarasını üretir.
    clusters = model.fit_predict(X_scaled)

    print("\nModelin Oluşturduğu Gruplar:")

    for i, customer in enumerate(X):
        income = customer[0]
        spending = customer[1]
        cluster = clusters[i]

        print(
            f"{i + 1}. Müşteri : "
            f"Gelir = {income}, Harcama = {spending} "
            f"--> Cluster = {cluster}"
        )

if __name__== "__main__":
    main()