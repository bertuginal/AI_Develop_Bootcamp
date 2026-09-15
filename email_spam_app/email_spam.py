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

class AppState:
    def __init__(self):
        self.csv_path: Optional[Path] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

        self.target_column: Optional[str] = None
        self.feature_columns: List[str] = None

        self.model_results: List[Dict[str, Any]] = []

        self.preprocessing_completed: bool = False
        self.current_step: int = 1