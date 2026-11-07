# eval_groupkfold.py - 투수 단위 GroupKFold(5) 재평가 (평가 전용, 배포 모델과 무관)
#
# 용도: README "검증" 단락과 streamlit_app/app.py의 성능 수치(weighted F1 0.859 ± 0.003,
#       정확도 0.860 ± 0.003)를 낸 스크립트. 투구 단위 랜덤 분할(desktop_app/src/train.py,
#       weighted F1 0.93)은 같은 투수가 학습·테스트에 섞이므로, 투수 ID(pitcher)로 묶어
#       처음 보는 투수 기준 성능을 잰다. 재학습 후 수치를 갱신할 때 다시 실행한다.
# 설정: 2024+2025 정규시즌, desktop_app/src의 전처리·17개 피처·MLP(build_model) 그대로,
#       seed 42, batch 512, EarlyStopping(patience 10). fold마다 학습 투수의 1/8을 투수 단위로
#       떼어 val로 사용(약 70/10/20).
# 필요 데이터: pybaseball 일별 캐시(~/.pybaseball/cache/get_statcast_data_from_csv_url*.parquet —
#       pybaseball.cache.enable() 상태로 2024·2025 statcast()를 받아 두면 생김). 원본 Statcast의
#       pitcher(투수 ID) 컬럼이 필요하다. data_loader.filter_and_clean()은 이 컬럼을 버리므로
#       desktop_app/data/*.csv 캐시로는 안 된다.
# 실행: python scripts/eval_groupkfold.py
#       환경변수 PYBASEBALL_CACHE_GLOB(캐시 경로 패턴), EVAL_OUT_DIR(결과 폴더, 기본 ./eval_results)
# 출력: EVAL_OUT_DIR/gkf_fold{0..4}.json, gkf_summary.json. fold별로 저장해 중단 후 재실행하면 이어서 돈다.
# 저장 코드 없음: ModelCheckpoint·pickle을 쓰지 않으며 desktop_app/model·backend/model을 덮어쓰지 않는다.
#
# Evaluation-only GroupKFold(5) by pitcher. Same data/preprocessing/17 features/MLP/seed as eval_by_pitcher.py.
# No ModelCheckpoint, no pickles; results only in EVAL_OUT_DIR. Each fold saved -> rerun resumes.
import sys, glob, os, json, gc
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'desktop_app', 'src'))
from data_loader import FEATURE_COLS, TARGET_PITCHES, LABEL_COL
from model import build_model
from tensorflow import keras
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

SEED = 42
OUT = os.environ.get('EVAL_OUT_DIR', 'eval_results')
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.expanduser(os.environ.get('PYBASEBALL_CACHE_GLOB', '~/.pybaseball/cache/get_statcast_data_from_csv_url*.parquet'))

cols = FEATURE_COLS + [LABEL_COL, 'pitcher', 'game_date']
df = pd.concat([pd.read_parquet(f, columns=cols) for f in glob.glob(CACHE)], ignore_index=True)
df = df[pd.to_datetime(df.game_date).dt.year.isin([2024, 2025])]
df = df[df[LABEL_COL].isin(TARGET_PITCHES)][FEATURE_COLS + [LABEL_COL, 'pitcher']].dropna()
df = df[(df.release_speed >= 40) & (df.release_speed <= 110)]
print('rows', len(df), 'pitchers', df.pitcher.nunique(), flush=True)

X = df[FEATURE_COLS].values.astype(np.float32)
le = LabelEncoder(); y = le.fit_transform(df[LABEL_COL].values)
g = df.pitcher.values
del df; gc.collect()
idx = np.arange(len(y))

# deterministic fold assignment (GroupKFold without shuffle). Val for EarlyStopping = 1/8 of the
# train-fold pitchers (also grouped) -> ~70/10/20 by pitcher.
for k, (trv, te) in enumerate(GroupKFold(n_splits=5).split(idx, y, g)):
    path = os.path.join(OUT, f'gkf_fold{k}.json')
    if os.path.exists(path):
        continue
    a, b = next(GroupShuffleSplit(1, test_size=0.125, random_state=SEED).split(trv, y[trv], g[trv]))
    tr, va = trv[a], trv[b]

    keras.utils.set_random_seed(SEED)
    sc = StandardScaler().fit(X[tr])
    m = build_model(X.shape[1], len(le.classes_))
    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)]
    m.fit(sc.transform(X[tr]), y[tr], validation_data=(sc.transform(X[va]), y[va]),
          epochs=100, batch_size=512, callbacks=cb, verbose=2)
    p = np.argmax(m.predict(sc.transform(X[te]), verbose=0), axis=1)
    rep = classification_report(y[te], p, target_names=le.classes_, output_dict=True)
    res = {'fold': k, 'n_test': int(len(te)), 'test_pitchers': int(len(set(g[te]))),
           'overlap_train_test': int(len(set(g[tr]) & set(g[te]))),
           'acc': accuracy_score(y[te], p), 'weighted_f1': f1_score(y[te], p, average='weighted'),
           'per_class_f1': {c: rep[c]['f1-score'] for c in le.classes_},
           'per_class_support': {c: int(rep[c]['support']) for c in le.classes_}}
    json.dump(res, open(path, 'w'), indent=1)
    print('FOLD', json.dumps(res), flush=True)

    del m, sc, p
    keras.backend.clear_session(); gc.collect()

# summary
rs = [json.load(open(os.path.join(OUT, f'gkf_fold{k}.json'))) for k in range(5)]
summ = {'weighted_f1': [r['weighted_f1'] for r in rs]}
for c in le.classes_:
    summ[c] = [r['per_class_f1'][c] for r in rs]
summary = {key: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)), 'folds': v} for key, v in summ.items()}
json.dump(summary, open(os.path.join(OUT, 'gkf_summary.json'), 'w'), indent=1)
print('SUMMARY', json.dumps(summary, indent=1))
