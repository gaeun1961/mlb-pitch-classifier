# wake_app.py - Streamlit Cloud 앱이 잠들어 있으면 깨우고, 앱 본문이 뜰 때까지 기다린다.
# GitHub Actions(keep_awake.yml)와 README 스크린샷 스크립트(take_screenshots.py)가 함께 쓴다.
# 실패(타임아웃)하면 0이 아닌 종료 코드로 끝나 job이 실패로 표시된다.

import os
import sys
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    'APP_URL', 'https://mlb-pitch-classifier-tgtrdgldkletpsw43j26sz.streamlit.app/')
WAKE_BUTTON = 'button:has-text("get this app back up"), button:has-text("Wake up")'
APP_READY = 'text=모델 정보'  # 앱 탭 바가 렌더되면 나타나는 텍스트


def open_app(page, timeout_ms=300_000):
    """앱 URL을 열고, sleep 화면이면 깨운 뒤, 앱 본문이 있는 frame을 반환한다."""
    page.goto(APP_URL, timeout=120_000)
    page.wait_for_load_state('networkidle', timeout=120_000)

    wake = page.locator(WAKE_BUTTON)
    if wake.count() and wake.first.is_visible():
        print('[wake] 잠든 앱 감지 → 깨우기 버튼 클릭')
        wake.first.click()

    # 앱 본문은 Streamlit Cloud 래퍼 페이지 안의 iframe(/~/+/)에서 렌더된다.
    app = page.frame_locator('iframe[title="streamlitApp"]')
    app.locator(APP_READY).first.wait_for(timeout=timeout_ms)
    print('[wake] 앱 본문 로드 확인')
    return app


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            open_app(page)
        except Exception as e:
            print(f'[wake] 실패: {e}', file=sys.stderr)
            page.screenshot(path='wake_failure.png', full_page=True)
            sys.exit(1)
        finally:
            browser.close()


if __name__ == '__main__':
    main()
