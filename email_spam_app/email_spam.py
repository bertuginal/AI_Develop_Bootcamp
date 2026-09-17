"""
=== Machine Learning(ML) Project: E-Mail Spam Prediction ===
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from colorama import Fore, Style, init as colorama_init

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


class AppState:
    def __init__(self):
        self.csv_path: Optional[Path] = None  # Seçilen orijinal CSV dosyasının bilgisayardaki yolunu tutar.
        self.raw_df: Optional[pd.DataFrame] = None  # Dosyadan yüklenen ilk, ham veri tablosudur.
        self.df: Optional[pd.DataFrame] = None  # İşlenmiş ve analizde kullanılacak güncel veri tablosudur.

        self.target_column: Optional[str] = None  # Modelin tahmin edeceği hedef sütunun adıdır (y).
        self.feature_columns: List[str] = None  # Eğitim için kullanılacak girdi sütunlarının listesidir (X).

        self.best_model: Optional[Pipeline] = None  # Eğitilenler arasından en yüksek başarıyı gösteren model nesnesidir.
        self.best_model_name: Optional[str] = None  # En başarılı modelin adını (örneğin "Random Forest") saklar.

        self.X_test: Optional[pd.DataFrame] = None  # Modeli test etmek için ayrılan bağımsız değişken (girdi) verileridir.
        self.y_test: Optional[pd.Series] = None  # Modeli test etmek için ayrılan gerçek hedef (çıktı) değerleridir.
        self.y_pred: Optional[np.ndarray] = None  # Modelin test verileri üzerinde yaptığı tahmin sonuçlarıdır.

        self.model_results: List[Dict[str, Any]] = []  # Eğitilen tüm modellerin başarı metriklerini saklayan listedir.

        self.preprocessing_completed: bool = False  # Veri ön işlemenin bitip bitmediğini gösteren onay işaretidir.
        self.current_step: int = 1  # Uygulamanın hangi işlem adımında olduğunu gösteren sayıdır.

        self.active_data_source: Optional[str] = None  # Şu an aktif olarak kullanılan veri kaynağının türünü veya adını belirtir.
        self.cleaned_csv_path: Optional[Path] = None  # Temizlenmiş ve işlenmiş verinin kaydedildiği yeni dosya yoludur.
        self.last_pdf_report_path: Optional[Path] = None  # Oluşturulan en son PDF analiz raporunun bilgisayardaki yoludur.

# Console ekranı daha okunabilir hale gelmesini sağlar.
def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

# Menü kullanıcının sonucu okuyabilmesi için ENTER tuşuna basmasını sağlar.
def pause() -> None:
    input("\nDevam etmek için lütfen ENTER tuşuna basınız...")

# Menü yazısını farklı renklerde kullanmamızı sağlar.
def print_menu_option(text: str) -> None:
    print(Fore.LIGHTCYAN_EX + text + Style.RESET_ALL)