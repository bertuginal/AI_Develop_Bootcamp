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


# STEP başlıklarını menü seçeneklerinden ayırmak için parlak camgöbeği renkte gösterir.
def print_step_title(text: str) -> None:
    print(Fore.CYAN + Style.BRIGHT + text + Style.RESET_ALL)


# CSV sütun adlarını daha düzenli hale getirmemizi sağlar. (KeliME Sayisı -> kelime_sayisi)
def normalize_column_name(name: str) -> str:
    value = str(name).replace("\ufeff", "").strip().lower()

    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("__")


# Kullanıcının dosya seçebilmesi için CSV dosyalarını tarar ve sadece görünen CSV dosyalarını eklemeye (dinamik olarak seçmeye) yarar.
def discover_cvs_files() -> List[Path]:
    found: List[Path] = []

    search_dirs = [
        Path.cwd(),
        Path.cwd() / "data"
    ]

    for folder in search_dirs:
        if not folder.exists() or not folder.is_dir():
            continue

        for file_path in folder.glob("*.csv"):
            resolved = file_path.resolve()
            if resolved not in found:
                found.append(resolved)
    return sorted(found, key = lambda p:p.name.lower())


# Manuel olarak (Copy/Paste) veya dinamik olarak girilen dosya yolunu seçmek için kullanılır.
def choose_csv_path() -> Optional[Path]:
    print_header("CSV DOSYASINI SEÇ")
    print_menu_option("0 - Ana menü")
    print_menu_option("1 - Dosya yolunu yaz")
    print_menu_option("2 - Dosya yolunu seç")

    choice = input("\nSeçiminiz: ").strip()
    if choice == "0":
        return None

    if choice == "1":
        raw_path = input(
            "\nCSV dosyasının tam yolunu giriniz: "
        ).strip().strip('"')

        if not raw_path:
            print("\nHATA: Dosya yolu boş bırakılamaz!")
            return None

        #expanduser: Kullanıcının ana dizinini otomatik olarak gerçek klasör yoluna çevirmeye yarar.
        path = Path(raw_path).expanduser()

        if not path.exists():
            print("\nHATA: Girilen dosya yolu bulunamadı!")
            return None

        if not path.is_file():
            print("\nHATA: Girilen yol doysa değil!")
            return None

        if not path.suffix.lower() != ".csv":
            print("\nHATA: Girilen dosya CSV uzantılı değil!")
            return None

        print(f"\nSeçilen CSV Dosyası:\n{path.resolve()}")
        return path.resolve()

    if choice == "2":
        # Kullanıcının fare ile dosya seçerek programa aktarılmasıdır.
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()

            # Dosya seçme penceresinin arkada kalmasını engellemeye çalışır.
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            selected_file = filedialog.askopenfilename(
                title="CSV Dosyasını Seç",
                filetypes=[
                    ("CSV Dosyaları", "*.csv"),
                    ("Tüm Dosyalar", "*"),
                ]
            )

            root.destroy()

            if not selected_file:
                print("\nDosya seçimi iptal edildi!")
                return None

            path = Path(selected_file)
            if not path.exists():
                print("\nHATA: Seçilen dosya bulunamadı!")
                return None

            if not path.suffix.lower() != ".csv":
                print("\nHATA: Lütfen CSV uzantılı bir dosya seçiniz!")
                return None

            print(f"\nSeçilen CSV Dosyası:\n{path.resolve()}")
            return path.resolve()

        except ImportError:
            print(
                "\nHATA: Bu python sürümünde 'tkinter' kütüohanesi bulunamadı!"
                "Alternatif olarak (1) dosya yolunu manuel olarak girebilirsiniz."
            )
            return None

    print("\nHATA: 0 ≤ X ≤ 2 arasında tam sayı seçmelisiniz.")
    return None








