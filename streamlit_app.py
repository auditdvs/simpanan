import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path('models')
DATA_COLUMNS = [
    'ID', 'NAMA', 'CENTER', 'KELOMPOK', 'HARI', 'JAM', 'SL', 'TRANS. DATE',
    'Db Qurban', 'Cr Qurban', 'Db Khusus', 'Cr Khusus',
    'Db HariRaya', 'Cr HariRaya', 'Db Pensiun', 'Cr Pensiun',
    'Db Pokok', 'Cr Pokok', 'Db SIPADAN', 'Cr SIPADAN',
    'Db Sukarela', 'Cr Sukarela', 'Db Wajib', 'Cr Wajib',
    'Db Total', 'Cr Total'
]


def detect_delimiter(buffer: io.BytesIO) -> str:
    pos = buffer.tell()
    first_line = buffer.readline().decode('utf-8', errors='ignore')
    buffer.seek(pos)
    return ';' if first_line.count(';') > first_line.count(',') else ','


def load_models() -> Tuple[object, object, Dict]:
    scaler_path = MODELS_DIR / 'scaler.pkl'
    iso_path = MODELS_DIR / 'isolation_forest.pkl'
    meta_path = MODELS_DIR / 'metadata.json'
    if not scaler_path.exists() or not iso_path.exists():
        st.error("Model tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
        st.stop()
    scaler = joblib.load(scaler_path)
    iso = joblib.load(iso_path)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return scaler, iso, metadata


def load_data(uploaded) -> pd.DataFrame:
    sep = detect_delimiter(uploaded)
    df_raw = pd.read_csv(uploaded, sep=sep)
    return df_raw


def clean_and_prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw[DATA_COLUMNS].copy()
    df['JAM'] = df['JAM'].astype(str).str.replace("'", "", regex=False)
    df['TRANS. DATE'] = pd.to_datetime(df['TRANS. DATE'], format='%d/%m/%Y')
    df['MINGGU'] = df['TRANS. DATE'].dt.isocalendar().week
    df['TAHUN'] = df['TRANS. DATE'].dt.year
    df['YEAR_WEEK'] = df['TAHUN'].astype(str) + '-W' + df['MINGGU'].astype(str).str.zfill(2)
    return df


def rolling_zscore(df: pd.DataFrame, window: int = 3, threshold: float = 1.0) -> pd.DataFrame:
    weekly = df.groupby(['ID', 'NAMA', 'CENTER', 'YEAR_WEEK']).agg({
        'Db HariRaya': 'sum',
        'TRANS. DATE': 'first'
    }).reset_index()
    weekly = weekly.sort_values(['ID', 'YEAR_WEEK'])

    def calc(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values('YEAR_WEEK').copy()
        g['Rolling_Mean'] = g['Db HariRaya'].rolling(window=window, min_periods=1).mean()
        g['Rolling_Std'] = g['Db HariRaya'].rolling(window=window, min_periods=1).std().fillna(0)
        g['Z_Score'] = np.where(
            g['Rolling_Std'] > 0,
            (g['Db HariRaya'] - g['Rolling_Mean']) / g['Rolling_Std'],
            0
        )
        g['Anomaly_HariRaya'] = (np.abs(g['Z_Score']) > threshold).astype(int)
        return g

    result = weekly.groupby('ID', group_keys=False).apply(calc)
    return result


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


def predict_sukarela(df: pd.DataFrame, scaler, iso, feature_cols: List[str]) -> pd.DataFrame:
    agg = aggregate_sukarela(df)
    if not feature_cols:
        feature_cols = [
            'Db Sukarela_Total', 'Db Sukarela_Avg', 'Db Sukarela_Std', 'Db Sukarela_Max',
            'Cr Sukarela_Total', 'Cr Sukarela_Avg', 'Cr Sukarela_Std', 'Cr Sukarela_Max'
        ]
    X = agg[feature_cols].values
    X_scaled = scaler.transform(X)
    agg['Anomaly_Sukarela_Score'] = iso.decision_function(X_scaled)
    agg['Anomaly_Sukarela'] = (iso.predict(X_scaled) == -1).astype(int)
    return agg


def combine_results(df_clean: pd.DataFrame, hari: pd.DataFrame, sukarela: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Anomali per ID
    hari_ids = hari[hari['Anomaly_HariRaya'] == 1][['ID']].drop_duplicates().copy()
    hari_ids['Is_Anomaly_HariRaya'] = 1

    sukarela_ids = sukarela[['ID', 'Anomaly_Sukarela']].rename(columns={'Anomaly_Sukarela': 'Is_Anomaly_Sukarela'})

    combined = df_clean[['ID', 'NAMA', 'CENTER']].drop_duplicates()
    combined = combined.merge(hari_ids, on='ID', how='left')
    combined = combined.merge(sukarela_ids[['ID', 'Is_Anomaly_Sukarela']], on='ID', how='left')
    combined['Is_Anomaly_HariRaya'] = combined['Is_Anomaly_HariRaya'].fillna(0).astype(int)
    combined['Is_Anomaly_Sukarela'] = combined['Is_Anomaly_Sukarela'].fillna(0).astype(int)
    combined['Anomaly_Final'] = ((combined['Is_Anomaly_HariRaya'] == 1) | (combined['Is_Anomaly_Sukarela'] == 1)).astype(int)

    rekap = combined.groupby('CENTER').agg({
        'ID': 'count',
        'Is_Anomaly_HariRaya': 'sum',
        'Is_Anomaly_Sukarela': 'sum',
        'Anomaly_Final': 'sum'
    }).reset_index()
    rekap.columns = ['CENTER', 'Total_Member', 'Anomali_HariRaya', 'Anomali_Sukarela', 'Total_Anomali_Final']
    rekap['Pct_Anomali'] = (rekap['Total_Anomali_Final'] / rekap['Total_Member'] * 100).round(2)
    rekap = rekap.sort_values('Total_Anomali_Final', ascending=False)
    return combined, rekap


def make_excel(anomaly_detail: pd.DataFrame, rekap: pd.DataFrame, combined: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        anomaly_detail.to_excel(writer, sheet_name='anomaly_detail', index=False)
        rekap.to_excel(writer, sheet_name='rekap_center', index=False)
        combined.to_excel(writer, sheet_name='data_with_flags', index=False)
    output.seek(0)
    return output.read()


def plot_top_centers(rekap: pd.DataFrame, mode: str):
    if mode == 'HariRaya':
        data = rekap[['CENTER', 'Anomali_HariRaya']].rename(columns={'Anomali_HariRaya': 'Anomali'})
    elif mode == 'Sukarela':
        data = rekap[['CENTER', 'Anomali_Sukarela']].rename(columns={'Anomali_Sukarela': 'Anomali'})
    else:
        data = rekap[['CENTER', 'Total_Anomali_Final']].rename(columns={'Total_Anomali_Final': 'Anomali'})
    data = data.head(15)
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Anomali:Q'),
        y=alt.Y('CENTER:N', sort='-x'),
        tooltip=['CENTER', 'Anomali']
    ).properties(height=400)
    return chart


def plot_breakdown(rekap: pd.DataFrame):
    top = rekap.head(15)
    melted = top.melt(id_vars=['CENTER'], value_vars=['Anomali_HariRaya', 'Anomali_Sukarela'], var_name='Jenis', value_name='Jumlah')
    chart = alt.Chart(melted).mark_bar().encode(
        x=alt.X('Jumlah:Q'),
        y=alt.Y('CENTER:N', sort='-x'),
        color=alt.Color('Jenis:N'),
        tooltip=['CENTER', 'Jenis', 'Jumlah']
    ).properties(height=400)
    return chart


def main():
    st.set_page_config(page_title='THC Anomaly Detector', layout='wide')
    st.title('THC Anomaly Detector')
    st.markdown('01. Ambil data dari database lalu rename jadi **THC.csv**,')
    st.markdown('02. Upload file tersebut, untuk detailnya ada dalam bentuk file excel.')
    st.markdown('03. Jika file nya di olah manual, ubah bagian header menajadi seperti ini:')
    st.markdown('ID,NAMA,CENTER,KELOMPOK,HARI,JAM,SL,TRANS. DATE,Db Qurban,Cr Qurban,Db Khusus,Cr Khusus,Db HariRaya,Cr HariRaya,Db Pensiun,Cr Pensiun,Db Pokok,Cr Pokok,Db SIPADAN,Cr SIPADAN,Db Sukarela,Cr Sukarela,Db Wajib,Cr Wajib,Db Total,Cr Total,Db PTN,Cr PTN,Db PRT,Cr PRT,Db DTP,Cr DTP,Db PMB,Cr PMB,Db PRR,Cr PRR,Db PSA,Cr PSA,Db PU,Cr PU,Db Total2,Cr Total2')

    scaler, iso, metadata = load_models()
    threshold = metadata.get('rolling_zscore_threshold', 1.0)

    uploaded = st.file_uploader('Upload file THC (CSV)', type=['csv'])
    mode_filter = st.selectbox('Tampilkan anomali untuk', ['HariRaya', 'Sukarela', 'Combined'])

    if not uploaded:
        st.info('Silakan upload file THC untuk mulai analisis.')
        return

    df_raw = load_data(uploaded)
    try:
        df_clean = clean_and_prepare(df_raw)
    except Exception as exc:
        st.error(f'Gagal memproses data: {exc}')
        st.stop()

    hari = rolling_zscore(df_clean, window=3, threshold=threshold)
    sukarela = predict_sukarela(df_clean, scaler, iso, metadata.get('feature_cols', []))
    combined, rekap = combine_results(df_clean, hari, sukarela)

    anomaly_detail = combined[combined['Anomaly_Final'] == 1].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Member', f"{len(combined):,}")
    col2.metric('Anomali HariRaya', f"{combined['Is_Anomaly_HariRaya'].sum():,}")
    col3.metric('Anomali Sukarela', f"{combined['Is_Anomaly_Sukarela'].sum():,}")
    col4.metric('Anomali Final', f"{combined['Anomaly_Final'].sum():,}")

    st.subheader('Unduh Hasil')
    excel_bytes = make_excel(anomaly_detail, rekap, combined)
    st.download_button('Download Excel (3 sheet)', excel_bytes, file_name='thc_anomaly_results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    st.subheader('Visualisasi')
    chart = plot_top_centers(rekap, mode_filter)
    st.altair_chart(chart, use_container_width=True)

    st.subheader('Breakdown HariRaya vs Sukarela (Top 15 CENTER)')
    st.altair_chart(plot_breakdown(rekap), use_container_width=True)

    st.subheader('Tabel Anomali')
    if mode_filter == 'HariRaya':
        table = hari[hari['Anomaly_HariRaya'] == 1][['ID', 'NAMA', 'CENTER', 'YEAR_WEEK', 'Db HariRaya', 'Z_Score']]
        st.dataframe(table)
    elif mode_filter == 'Sukarela':
        table = sukarela[sukarela['Anomaly_Sukarela'] == 1][['ID', 'NAMA', 'CENTER', 'Db Sukarela_Total', 'Cr Sukarela_Total', 'Anomaly_Decision_Score']]
        st.dataframe(table)
    else:
        st.dataframe(anomaly_detail[['ID', 'NAMA', 'CENTER', 'Is_Anomaly_HariRaya', 'Is_Anomaly_Sukarela']])


if __name__ == '__main__':
    main()
