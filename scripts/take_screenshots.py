# take_screenshots.py - 배포 앱을 깨운 뒤 README용 스크린샷 2장을 docs/images/에 저장한다.
#   (a) 예측 결과 화면(대시보드, 앱 기본 입력값)  (b) 모델 정보 탭
# 화면 수치가 README와 같은지(86.0% / 0.859) 확인하고, 다르면 실패로 끝난다.
# 실행: pip install playwright && python -m playwright install chromium && python scripts/take_screenshots.py

import os
import sys
from playwright.sync_api import sync_playwright
from wake_app import APP_URL, APP_READY, open_app

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images')
VIEWPORT = {'width': 1200, 'height': 1400}   # README 이미지 가로 1200px 이하
EXPECTED = ['86.0%', '0.859']


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        open_app(page)  # 잠들어 있으면 깨운다

        # 래퍼(iframe) 없이 앱 본문만 직접 열어야 스크롤 영역 전체가 캡처된다.
        page.goto(APP_URL.rstrip('/') + '/~/+/', timeout=120_000)
        page.locator(APP_READY).first.wait_for(timeout=180_000)
        page.locator('.pw-summary').first.wait_for(timeout=120_000)  # 첫 예측 완료
        page.wait_for_timeout(5_000)  # Plotly 차트 렌더 여유

        page.screenshot(path=os.path.join(OUT_DIR, 'app_prediction.png'))

        page.get_by_role('tab', name='모델 정보').click()
        page.get_by_text('Weighted F1').first.wait_for()
        page.wait_for_timeout(3_000)
        text = page.inner_text('body')
        page.screenshot(path=os.path.join(OUT_DIR, 'app_model_info.png'))
        browser.close()

    missing = [v for v in EXPECTED if v not in text]
    if missing:
        sys.exit(f'[확인 실패] 모델 정보 탭에 {missing} 없음 — 배포 반영 여부 확인 필요')
    print(f'[완료] {OUT_DIR} 저장, 화면 수치 {EXPECTED} 확인')


if __name__ == '__main__':
    main()
