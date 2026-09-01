"""fetch_daily.py - 전날(KST 기준) MLB 경기 Statcast 데이터를 연간 CSV에 누적 저장한다.

GitHub Actions에서 매일 KST 새벽 6시(cron: UTC 21:00)에 실행된다.
"""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pybaseball import statcast

DATA_DIR = Path(__file__).parent.parent / "data"


def target_date():
    kst_now = datetime.now(ZoneInfo("Asia/Seoul"))
    return (kst_now - timedelta(days=1)).date()


def main():
    day = target_date()
    csv_path = DATA_DIR / f"statcast_{day.year}.csv"

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

    combined = pd.concat([existing, df], ignore_index=True) if existing is not None else df
    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(csv_path, index=False)
    print(f"{len(df)}건 추가 완료 (누적 {len(combined)}건) -> {csv_path}")


if __name__ == "__main__":
    main()
