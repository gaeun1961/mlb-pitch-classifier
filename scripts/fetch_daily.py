"""fetch_daily.py - 전날 MLB 경기 Statcast 데이터를 연간 CSV(gzip)에 누적 저장한다.

GitHub Actions에서 매일 KST 새벽 6시(cron: UTC 21:00)에 실행된다. 타겟 날짜는
KST가 아니라 UTC 기준 "실행 시점 날짜 - 1일"로 계산한다 — KST 06:00은 UTC로
전날 21:00이라 KST 기준으로 하루를 빼면 실제로는 UTC 기준 당일이 되어버리고,
서부 지역 야간 경기는 UTC 기준 다음날 새벽까지 이어지므로 그 날짜 경기가
아직 안 끝났거나 Statcast에 반영되지 않았을 수 있다. UTC 기준으로 하루 전을
잡으면 그 날짜의 모든 경기가 확실히 끝난 뒤이다.

Statcast는 원본 118개 컬럼을 주지만 모델 재학습(FEATURE_COLS + pitch_type)과
검색 UI 표시(player_name/p_throws/game_date)에 쓰는 것만 남긴다 — 시즌 전체를
누적하면 원본 그대로는 GitHub의 파일당 100MB 제한을 훌쩍 넘어서기 때문이다
(2026-03-25~09-01만 해도 원본 367MB, 컬럼을 추리고 gzip하면 59MB).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from pybaseball import statcast

DATA_DIR = Path(__file__).parent.parent / "data"

KEEP_COLS = [
    'pitch_type', 'game_date', 'player_name', 'p_throws',
    'release_speed', 'release_spin_rate', 'release_extension',
    'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'effective_speed', 'spin_axis',
]


def target_date():
    return (datetime.now(timezone.utc) - timedelta(days=1)).date()


def main():
    day = target_date()
    csv_path = DATA_DIR / f"statcast_{day.year}.csv.gz"

    existing = None
    if csv_path.exists():
        existing = pd.read_csv(csv_path, low_memory=False)
        if "game_date" in existing.columns and str(day) in existing["game_date"].astype(str).values:
            print(f"{day} 데이터는 이미 있습니다. 스킵합니다.")
            return

    print(f"{day} Statcast 데이터를 가져오는 중...")
    df = statcast(str(day), str(day))

    if df.empty:
        print(f"{day}에는 경기 기록이 없습니다 (시즌 오프이거나 휴식일). 종료합니다.")
        return

    df = df[[c for c in KEEP_COLS if c in df.columns]]
    combined = pd.concat([existing, df], ignore_index=True) if existing is not None else df
    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(csv_path, index=False)
    print(f"{len(df)}건 추가 완료 (누적 {len(combined)}건) -> {csv_path}")


if __name__ == "__main__":
    main()
