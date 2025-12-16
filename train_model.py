import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path('THC.csv')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    'Db Sukarela_Total', 'Db Sukarela_Avg', 'Db Sukarela_Std', 'Db Sukarela_Max',
    'Cr Sukarela_Total', 'Cr Sukarela_Avg', 'Cr Sukarela_Std', 'Cr Sukarela_Max'
]

ISOFOREST_PARAMS = dict(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)


def detect_delimiter(file_path: Path) -> str:
    with file_path.open('r', encoding='utf-8') as f:
        first_line = f.readline()
    return ';' if first_line.count(';') > first_line.count(',') else ','


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    sep = detect_delimiter(path)
    df_raw = pd.read_csv(path, sep=sep)
    return df_raw


def clean_and_prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    columns_needed = [
        'ID', 'NAMA', 'CENTER', 'KELOMPOK', 'HARI', 'JAM', 'SL', 'TRANS. DATE',
        'Db Qurban', 'Cr Qurban', 'Db Khusus', 'Cr Khusus',
        'Db HariRaya', 'Cr HariRaya', 'Db Pensiun', 'Cr Pensiun',
        'Db Pokok', 'Cr Pokok', 'Db SIPADAN', 'Cr SIPADAN',
        'Db Sukarela', 'Cr Sukarela', 'Db Wajib', 'Cr Wajib',
        'Db Total', 'Cr Total'
    ]
    df = df_raw[columns_needed].copy()
    df['JAM'] = df['JAM'].astype(str).str.replace("'", "", regex=False)
    df['TRANS. DATE'] = pd.to_datetime(df['TRANS. DATE'], format='%d/%m/%Y')
    df['MINGGU'] = df['TRANS. DATE'].dt.isocalendar().week
    df['TAHUN'] = df['TRANS. DATE'].dt.year
    df['YEAR_WEEK'] = df['TAHUN'].astype(str) + '-W' + df['MINGGU'].astype(str).str.zfill(2)
    return df


def aggregate_sukarela(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(['ID', 'NAMA', 'CENTER']).agg({
        'Db Sukarela': ['sum', 'mean', 'std', 'max', 'count'],
        'Cr Sukarela': ['sum', 'mean', 'std', 'max'],
    }).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    agg.columns = agg.columns.str.replace('sum', 'Total')
    agg.columns = agg.columns.str.replace('mean', 'Avg')
    agg.columns = agg.columns.str.replace('std', 'Std')
    agg.columns = agg.columns.str.replace('max', 'Max')
    agg.columns = agg.columns.str.replace('count', 'Count')
    agg = agg.fillna(0)
    return agg


def train_and_save(df: pd.DataFrame) -> None:
    agg = aggregate_sukarela(df)
    X = agg[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(**ISOFOREST_PARAMS)
    iso.fit(X_scaled)

    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(iso, MODELS_DIR / 'isolation_forest.pkl')

    metadata = {
        'feature_cols': FEATURE_COLS,
        'isolation_forest_params': ISOFOREST_PARAMS,
        'rolling_zscore_threshold': 1.0,
        'rolling_window': 3,
        'source_data': str(DATA_FILE),
    }
    (MODELS_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print("Model dan scaler berhasil disimpan di folder 'models/'")


def main():
    df_raw = load_data(DATA_FILE)
    df = clean_and_prepare(df_raw)
    train_and_save(df)


if __name__ == '__main__':
    main()
