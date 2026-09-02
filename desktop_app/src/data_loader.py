# data_loader.py - Statcast 데이터 수집 및 전처리

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from pybaseball import statcast, cache as pybaseball_cache
    pybaseball_cache.enable()  # 일별 하위 요청을 캐시해 다운로드 중 실패해도 재시도 비용을 줄인다
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("[경고] pybaseball이 설치되지 않았습니다.")

DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'model')
SCALER_PATH  = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# 학습에 사용할 시즌별 (시작일, 종료일). 완결된 시즌만 포함한다 — 진행 중인
# 시즌을 넣으면 그 시즌만 표본이 적어 구종 비율이 왜곡된다.
TRAIN_SEASONS = [
    ('2024', '2024-03-20', '2024-11-01'),
    ('2025', '2025-03-27', '2025-11-01'),
]

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_PITCHES = ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'FS']

FEATURE_COLS = [
    'release_speed',
    'release_spin_rate',
    'release_extension',
    'release_pos_x',
    'release_pos_z',
    'pfx_x',
    'pfx_z',
    'plate_x',
    'plate_z',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az',
    'effective_speed',
    'spin_axis',
]

LABEL_COL = 'pitch_type'


def download_statcast(year, start_date, end_date):
    """해당 시즌의 Statcast 데이터를 다운로드한다. 캐시 파일이 있으면 바로 로드한다."""
    cache_path = os.path.join(DATA_DIR, f'statcast_{year}.csv')
    if os.path.exists(cache_path):
        print(f"[정보] 캐시 파일 로드: {cache_path}")
        return pd.read_csv(cache_path, low_memory=False)

    if not PYBASEBALL_AVAILABLE:
        raise RuntimeError("pybaseball이 없어 데이터를 다운로드할 수 없습니다.")

    print(f"[정보] 다운로드 중: {start_date} ~ {end_date}")
    df = statcast(start_dt=start_date, end_dt=end_date)
    df.to_csv(cache_path, index=False)
    print(f"[정보] 저장 완료: {cache_path}  ({len(df):,}행)")
    return df


def download_all_seasons(seasons=TRAIN_SEASONS):
    """TRAIN_SEASONS에 지정된 시즌들을 모두 다운로드해 하나로 합친다."""
    dfs = [download_statcast(year, start, end) for year, start, end in seasons]
    return pd.concat(dfs, ignore_index=True)


def filter_and_clean(df):
    """대상 구종만 남기고 결측값과 구속 이상치(40~110mph)를 제거한다."""
    print(f"[전처리] 원본: {len(df):,}행")

    df = df[df[LABEL_COL].isin(TARGET_PITCHES)].copy()
    df = df[FEATURE_COLS + [LABEL_COL]]
    df = df.dropna()
    df = df[(df['release_speed'] >= 40) & (df['release_speed'] <= 110)]

    print(f"[전처리] 정제 후: {len(df):,}행")
    print(df[LABEL_COL].value_counts().to_string())

    return df


def encode_labels(df):
    """구종명을 정수로 인코딩하고 인코더를 저장한다."""
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL].values)
    label_map = {i: cls for i, cls in enumerate(le.classes_)}

    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)

    print(f"[레이블] {label_map}")
    return y, label_map, le


def normalize_features(X_train, X_val, X_test):
    """훈련 세트 기준으로 StandardScaler를 적합시키고 스케일러를 저장한다."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"[정규화] 스케일러 저장: {SCALER_PATH}")
    return X_train, X_val, X_test, scaler


def load_data(seasons=TRAIN_SEASONS, val_ratio=0.10, test_ratio=0.10, random_state=42):
    """다운로드부터 정규화까지 전체 전처리 파이프라인을 실행한다."""
    df = download_all_seasons(seasons)
    df = filter_and_clean(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y, label_map, _ = encode_labels(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_ratio + test_ratio,
        random_state=random_state, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state, stratify=y_temp
    )

    print(f"[분할] train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")

    X_train, X_val, X_test, _ = normalize_features(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_map


if __name__ == '__main__':
    X_tr, X_va, X_te, y_tr, y_va, y_te, lmap = load_data()
    print(f"\n특징 수: {X_tr.shape[1]}  구종 수: {len(lmap)}")
