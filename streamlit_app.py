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


# ── Helpers ──────────────────────────────────────────────────────────────────

def detect_delimiter(buffer: io.BytesIO) -> str:
    pos = buffer.tell()
    first_line = buffer.readline().decode('utf-8', errors='ignore')
    buffer.seek(pos)
    counts = {'\t': first_line.count('\t'), ';': first_line.count(';'), ',': first_line.count(',')}
    return max(counts, key=counts.get)


def load_models() -> Tuple[object, object, Dict]:
    scaler_path = MODELS_DIR / 'scaler.pkl'
    iso_path    = MODELS_DIR / 'isolation_forest.pkl'
    meta_path   = MODELS_DIR / 'metadata.json'
    if not scaler_path.exists() or not iso_path.exists():
        st.error("Model tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
        st.stop()
    scaler   = joblib.load(scaler_path)
    iso      = joblib.load(iso_path)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return scaler, iso, metadata


def load_data(uploaded) -> pd.DataFrame:
    sep = detect_delimiter(uploaded)
    return pd.read_csv(uploaded, sep=sep)


def clean_and_prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Hanya ambil kolom yang diperlukan
    missing = [c for c in DATA_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di file: {missing}")
    df = df_raw[DATA_COLUMNS].copy()
    df['JAM'] = df['JAM'].astype(str).str.replace("'", "", regex=False)
    df['TRANS. DATE'] = pd.to_datetime(df['TRANS. DATE'], format='%d/%m/%Y')
    df['MINGGU']    = df['TRANS. DATE'].dt.isocalendar().week.astype(int)
    df['TAHUN']     = df['TRANS. DATE'].dt.year
    df['YEAR_WEEK'] = df['TAHUN'].astype(str) + '-W' + df['MINGGU'].astype(str).str.zfill(2)
    return df


# ── Detection: HariRaya ───────────────────────────────────────────────────────

def rolling_zscore(df: pd.DataFrame, window: int = 3, threshold: float = 1.0) -> pd.DataFrame:
    weekly = (
        df.groupby(['ID', 'NAMA', 'CENTER', 'YEAR_WEEK'], as_index=False)
        .agg({'Db HariRaya': 'sum', 'TRANS. DATE': 'first'})
        .sort_values(['ID', 'YEAR_WEEK'])
        .reset_index(drop=True)
    )

    records = []
    for id_val, group in weekly.groupby('ID', sort=False):
        g = group.sort_values('YEAR_WEEK').copy()
        g['Rolling_Mean'] = g['Db HariRaya'].rolling(window=window, min_periods=1).mean()
        g['Rolling_Std']  = g['Db HariRaya'].rolling(window=window, min_periods=1).std().fillna(0)
        g['Z_Score'] = np.where(
            g['Rolling_Std'] > 0,
            (g['Db HariRaya'] - g['Rolling_Mean']) / g['Rolling_Std'],
            0
        )
        g['Anomaly_HariRaya'] = (np.abs(g['Z_Score']) > threshold).astype(int)
        records.append(g)

    result = pd.concat(records, ignore_index=True)
    return result


# ── Detection: Sukarela ───────────────────────────────────────────────────────

def aggregate_sukarela(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(['ID', 'NAMA', 'CENTER'], as_index=False)
        .agg(
            Db_Sukarela_Total  = ('Db Sukarela', 'sum'),
            Db_Sukarela_Avg    = ('Db Sukarela', 'mean'),
            Db_Sukarela_Std    = ('Db Sukarela', 'std'),
            Db_Sukarela_Max    = ('Db Sukarela', 'max'),
            Db_Sukarela_Count  = ('Db Sukarela', 'count'),
            Cr_Sukarela_Total  = ('Cr Sukarela', 'sum'),
            Cr_Sukarela_Avg    = ('Cr Sukarela', 'mean'),
            Cr_Sukarela_Std    = ('Cr Sukarela', 'std'),
            Cr_Sukarela_Max    = ('Cr Sukarela', 'max'),
        )
        .fillna(0)
    )
    return agg


DEFAULT_FEATURE_COLS = [
    'Db_Sukarela_Total', 'Db_Sukarela_Avg', 'Db_Sukarela_Std', 'Db_Sukarela_Max',
    'Cr_Sukarela_Total', 'Cr_Sukarela_Avg', 'Cr_Sukarela_Std', 'Cr_Sukarela_Max',
]


def predict_sukarela(df: pd.DataFrame, scaler, iso, feature_cols: List[str]) -> pd.DataFrame:
    agg = aggregate_sukarela(df)
    cols = feature_cols if feature_cols else DEFAULT_FEATURE_COLS
    # Pastikan semua feature ada
    cols = [c for c in cols if c in agg.columns]
    if not cols:
        cols = [c for c in DEFAULT_FEATURE_COLS if c in agg.columns]
    X        = agg[cols].values
    X_scaled = scaler.transform(X)
    agg['Anomaly_Sukarela_Score'] = iso.decision_function(X_scaled)
    agg['Anomaly_Sukarela']       = (iso.predict(X_scaled) == -1).astype(int)
    return agg


# ── Combine ───────────────────────────────────────────────────────────────────

def combine_results(
    df_clean: pd.DataFrame,
    hari: pd.DataFrame,
    sukarela: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    hari_ids = (
        hari.loc[hari['Anomaly_HariRaya'] == 1, 'ID']
        .drop_duplicates()
        .to_frame()
    )
    hari_ids['Is_Anomaly_HariRaya'] = 1

    sukarela_ids = sukarela[['ID', 'Anomaly_Sukarela']].rename(
        columns={'Anomaly_Sukarela': 'Is_Anomaly_Sukarela'}
    )

    combined = df_clean[['ID', 'NAMA', 'CENTER']].drop_duplicates().reset_index(drop=True)
    combined = combined.merge(hari_ids,     on='ID', how='left')
    combined = combined.merge(sukarela_ids, on='ID', how='left')
    combined['Is_Anomaly_HariRaya'] = combined['Is_Anomaly_HariRaya'].fillna(0).astype(int)
    combined['Is_Anomaly_Sukarela'] = combined['Is_Anomaly_Sukarela'].fillna(0).astype(int)
    combined['Anomaly_Final'] = (
        (combined['Is_Anomaly_HariRaya'] == 1) | (combined['Is_Anomaly_Sukarela'] == 1)
    ).astype(int)

    rekap = (
        combined.groupby('CENTER', as_index=False)
        .agg(
            Total_Member          = ('ID', 'count'),
            Anomali_HariRaya      = ('Is_Anomaly_HariRaya', 'sum'),
            Anomali_Sukarela      = ('Is_Anomaly_Sukarela', 'sum'),
            Total_Anomali_Final   = ('Anomaly_Final', 'sum'),
        )
    )
    rekap['Pct_Anomali'] = (rekap['Total_Anomali_Final'] / rekap['Total_Member'] * 100).round(2)
    rekap = rekap.sort_values('Total_Anomali_Final', ascending=False).reset_index(drop=True)
    return combined, rekap


# ── Excel export ──────────────────────────────────────────────────────────────

def make_excel(
    anomaly_detail: pd.DataFrame,
    rekap: pd.DataFrame,
    combined: pd.DataFrame,
    hari_detail: pd.DataFrame,
    sukarela_detail: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()

    hari_anomali_weeks = hari_detail.loc[hari_detail['Anomaly_HariRaya'] == 1, ['ID', 'YEAR_WEEK']]
    hari_trans = (
        df_clean.merge(hari_anomali_weeks, on=['ID', 'YEAR_WEEK'], how='inner')
        [['ID', 'NAMA', 'CENTER', 'TRANS. DATE', 'YEAR_WEEK', 'Db HariRaya']]
        .sort_values(['ID', 'TRANS. DATE'])
        .merge(hari_detail[['ID', 'YEAR_WEEK', 'Z_Score', 'Anomaly_HariRaya']], on=['ID', 'YEAR_WEEK'], how='left')
    )

    sukarela_anomali_ids = sukarela_detail.loc[sukarela_detail['Anomaly_Sukarela'] == 1, 'ID'].unique()
    sukarela_trans = (
        df_clean[df_clean['ID'].isin(sukarela_anomali_ids)]
        [['ID', 'NAMA', 'CENTER', 'TRANS. DATE', 'Db Sukarela', 'Cr Sukarela']]
        .sort_values(['ID', 'TRANS. DATE'])
        .merge(sukarela_detail[['ID', 'Anomaly_Sukarela', 'Anomaly_Sukarela_Score']], on='ID', how='left')
    )

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        anomaly_detail.to_excel(writer, sheet_name='anomaly_summary',   index=False)
        rekap.to_excel(         writer, sheet_name='rekap_center',       index=False)
        hari_trans.to_excel(    writer, sheet_name='detail_hariraya',    index=False)
        sukarela_trans.to_excel(writer, sheet_name='detail_sukarela',    index=False)
        combined.to_excel(      writer, sheet_name='all_members_flags',  index=False)
    output.seek(0)
    return output.read()


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_top_centers(rekap: pd.DataFrame, mode: str):
    col_map = {
        'HariRaya': 'Anomali_HariRaya',
        'Sukarela': 'Anomali_Sukarela',
        'Combined': 'Total_Anomali_Final',
    }
    col  = col_map.get(mode, 'Total_Anomali_Final')
    data = rekap[['CENTER', col]].rename(columns={col: 'Anomali'}).head(15)
    return (
        alt.Chart(data).mark_bar()
        .encode(
            x=alt.X('Anomali:Q'),
            y=alt.Y('CENTER:N', sort='-x'),
            tooltip=['CENTER', 'Anomali'],
        )
        .properties(height=400)
    )


def plot_breakdown(rekap: pd.DataFrame):
    melted = (
        rekap.head(15)
        .melt(id_vars=['CENTER'], value_vars=['Anomali_HariRaya', 'Anomali_Sukarela'],
              var_name='Jenis', value_name='Jumlah')
    )
    return (
        alt.Chart(melted).mark_bar()
        .encode(
            x=alt.X('Jumlah:Q'),
            y=alt.Y('CENTER:N', sort='-x'),
            color=alt.Color('Jenis:N'),
            tooltip=['CENTER', 'Jenis', 'Jumlah'],
        )
        .properties(height=400)
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title='THC Anomaly Detector', layout='wide')
    st.title('THC Anomaly Detector')
    st.markdown('**Petunjuk:**')
    st.markdown('1. Ambil data dari database (Req Database OPTIMA) lalu rename jadi **THC.csv**')
    st.markdown('2. Upload file tersebut — hasil detail tersedia dalam bentuk file Excel')
    st.markdown('3. Jika file diolah manual, pastikan header sesuai format berikut:')
    st.code('ID | NAMA | CENTER | KELOMPOK | HARI | JAM | SL | TRANS. DATE | Db Qurban | Cr Qurban | '
            'Db Khusus | Cr Khusus | Db HariRaya | Cr HariRaya | Db Pensiun | Cr Pensiun | '
            'Db Pokok | Cr Pokok | Db SIPADAN | Cr SIPADAN | Db Sukarela | Cr Sukarela | '
            'Db Wajib | Cr Wajib | Db Total | Cr Total')

    scaler, iso, metadata = load_models()
    threshold = metadata.get('rolling_zscore_threshold', 1.0)

    uploaded    = st.file_uploader('Upload file THC (CSV)', type=['csv'])
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

    with st.spinner('Mendeteksi anomali...'):
        hari      = rolling_zscore(df_clean, window=3, threshold=threshold)
        sukarela  = predict_sukarela(df_clean, scaler, iso, metadata.get('feature_cols', []))
        combined, rekap = combine_results(df_clean, hari, sukarela)

    anomaly_detail = combined[combined['Anomaly_Final'] == 1].copy()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Member',      f"{len(combined):,}")
    col2.metric('Anomali HariRaya',  f"{combined['Is_Anomaly_HariRaya'].sum():,}")
    col3.metric('Anomali Sukarela',  f"{combined['Is_Anomaly_Sukarela'].sum():,}")
    col4.metric('Anomali Final',     f"{combined['Anomaly_Final'].sum():,}")

    # Download
    st.subheader('Unduh Hasil')
    excel_bytes = make_excel(anomaly_detail, rekap, combined, hari, sukarela, df_clean)
    st.download_button(
        'Download Excel (5 sheet)', excel_bytes,
        file_name='thc_anomaly_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    # Charts
    st.subheader('Visualisasi Top 15 CENTER')
    st.altair_chart(plot_top_centers(rekap, mode_filter), use_container_width=True)

    st.subheader('Breakdown HariRaya vs Sukarela (Top 15 CENTER)')
    st.altair_chart(plot_breakdown(rekap), use_container_width=True)

    # Detail table
    st.subheader('Tabel Detail Anomali')
    fmt_date = lambda df: df.assign(**{'TRANS. DATE': pd.to_datetime(df['TRANS. DATE']).dt.strftime('%Y-%m-%d')})

    if mode_filter == 'HariRaya':
        hari_anomali_weeks = hari.loc[hari['Anomaly_HariRaya'] == 1, ['ID', 'YEAR_WEEK']]
        detail_trans = (
            df_clean.merge(hari_anomali_weeks, on=['ID', 'YEAR_WEEK'], how='inner')
            [['ID', 'NAMA', 'CENTER', 'TRANS. DATE', 'YEAR_WEEK', 'Db HariRaya']]
            .sort_values(['ID', 'TRANS. DATE'])
            .merge(hari[['ID', 'YEAR_WEEK', 'Z_Score', 'Anomaly_HariRaya']], on=['ID', 'YEAR_WEEK'], how='left')
        )
        st.dataframe(fmt_date(detail_trans), use_container_width=True)

    elif mode_filter == 'Sukarela':
        sukarela_anomali_ids = sukarela.loc[sukarela['Anomaly_Sukarela'] == 1, 'ID'].unique()
        detail_trans = (
            df_clean[df_clean['ID'].isin(sukarela_anomali_ids)]
            [['ID', 'NAMA', 'CENTER', 'TRANS. DATE', 'Db Sukarela', 'Cr Sukarela']]
            .sort_values(['ID', 'TRANS. DATE'])
            .merge(sukarela[['ID', 'Anomaly_Sukarela', 'Anomaly_Sukarela_Score']], on='ID', how='left')
        )
        st.dataframe(fmt_date(detail_trans), use_container_width=True)

    else:
        anomaly_ids  = anomaly_detail['ID'].unique()
        detail_trans = (
            df_clean[df_clean['ID'].isin(anomaly_ids)]
            [['ID', 'NAMA', 'CENTER', 'TRANS. DATE', 'Db HariRaya', 'Db Sukarela', 'Cr Sukarela']]
            .sort_values(['ID', 'TRANS. DATE'])
        )
        st.dataframe(fmt_date(detail_trans), use_container_width=True)


if __name__ == '__main__':
    main()
