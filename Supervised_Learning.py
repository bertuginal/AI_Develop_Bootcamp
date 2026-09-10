"""
Bir öğrencinin:
1) Günlük çalışma saati
2) Derse katılım yüzdesi
bu bilgilere bakarak sınavı geçip geçmeyeceğini tahmin edelim.
Label:
    0: Kaldı
    1: Geçti
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    X = np.array([
        [1,30],
        [2,40],
        [2,50],
        [3,55],
        [4,60],
        [5,65],
        [6,75],
        [7,85],
        [8,90],
        [9,95]
    ])

    y = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1
    ])

    print("SUPERVISED LEARNING (Features(+) Label(+))")
    print("\nX - Öğrenci Özellikleri (Features)")
    print(X)

    print("\ny - Label(Etiketler)")
    print(y)



