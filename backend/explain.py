"""explain.py - Gemini API 기반 예측 결과 자연어 설명 생성 (실패 시 규칙 기반 폴백)"""

import os

import google.generativeai as genai

PITCH_NAMES = {
    'FF': '포심 패스트볼', 'SI': '싱커',    'SL': '슬라이더',
    'CU': '커브',          'CH': '체인지업', 'FC': '커터', 'FS': '스플리터',
}

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _model = genai.GenerativeModel('gemini-3.5-flash-lite')


def _runner_up(pred_label, proba_dict):
    return sorted(
        ((label, p) for label, p in proba_dict.items() if label != pred_label),
        key=lambda item: item[1], reverse=True,
    )[0]


def _ax_direction_desc(ax, p_throws):
    """앱의 '구종별 만들기 가이드'와 동일한 규칙: 우완 기준 ax 양수는 타자 바깥쪽으로
    휘는 움직임이고, 좌완은 좌우가 반대로 적용된다. 부호 해석을 여기서 미리 확정해
    Gemini가 손잡이별 부호를 직접 계산하다 틀리는 일이 없도록 한다."""
    if abs(ax) < 2:
        return "거의 휘지 않고 곧게 들어가는"
    is_away = (ax > 0) if p_throws != 'L' else (ax < 0)
    return "타자 바깥쪽으로 흘러나가는" if is_away else "타자 몸쪽을 파고드는"


def _build_prompt(input_dict, pred_label, confidence, proba_dict, p_throws):
    name = PITCH_NAMES.get(pred_label, pred_label)
    runner_up_label, runner_up_conf = _runner_up(pred_label, proba_dict)
    runner_up_name = PITCH_NAMES.get(runner_up_label, runner_up_label)
    side = "좌완" if p_throws == 'L' else "우완"
    ax_desc = _ax_direction_desc(input_dict['ax'], p_throws)

    return f"""당신은 MLB 중계 해설가입니다. 아래 투구 데이터를 보고 이 투구가 왜 해당 구종으로
분류되었는지, 그리고 2순위 구종이 아니라 왜 1순위 구종인지를 2~3문장의 자연스럽고 전문적인
한국어로 설명해주세요.

작성 규칙:
- 피처 이름(release_speed, az 등)이나 raw 수치를 그대로 나열하지 말고, 그 의미를 야구
  맥락으로 풀어서 설명하세요 (예: "묵직한 구속", "급격한 낙차", "타자 몸쪽을 파고드는 움직임").
- 1순위 구종과 2순위 구종을 비교하는 문장을 반드시 포함하세요.
- 매번 다른 문장 구조와 표현을 쓰세요. "~등의 특징을 종합하여 분류했습니다" 같은 상투적인
  틀은 피하세요.
- 이 투수의 투구 손(좌완/우완)을 고려해 움직임을 설명하세요.

투수: {side}
1순위 예측: {pred_label} ({name}), 신뢰도 {confidence * 100:.1f}%
2순위 예측: {runner_up_label} ({runner_up_name}), 확률 {runner_up_conf * 100:.1f}%

투구 데이터:
- 구속: {input_dict['release_speed']:.1f} mph (체감 구속 {input_dict['effective_speed']:.1f} mph)
- 수직 가속도(az): {input_dict['az']:.1f} (양수일수록 덜 떨어지는 포심 계열, 음수일수록
  급격히 가라앉는 변화구 계열)
- 수직 무브먼트(pfx_z): 약 {input_dict['pfx_z'] * 12:.1f}인치
- 수평 무브먼트(pfx_x): 약 {input_dict['pfx_x'] * 12:.1f}인치, {ax_desc} 궤적
- 회전수: {input_dict['release_spin_rate']:.0f} rpm

설명:"""


def _describe_speed(v):
    if v >= 93:
        return f"구속 {v:.1f}mph로 매우 빠른 편"
    if v >= 88:
        return f"구속 {v:.1f}mph로 준수한 편"
    if v >= 82:
        return f"구속 {v:.1f}mph로 중간 정도"
    return f"구속 {v:.1f}mph로 느린 편"


def _describe_az(v):
    if v >= 5:
        return f"수직 가속도 az={v:.1f}로 뚜렷한 양수(떠오르는 궤적)"
    if v >= -3:
        return f"수직 가속도 az={v:.1f}로 0 근처(거의 일직선에 가까운 궤적)"
    if v >= -20:
        return f"수직 가속도 az={v:.1f}로 음수(중간 정도 가라앉는 궤적)"
    return f"수직 가속도 az={v:.1f}로 큰 음수(급격히 떨어지는 궤적)"


def _describe_spin(v):
    if v >= 2400:
        return f"회전수 {v:.0f}rpm으로 높은 편(포심/커터 계열에서 흔함)"
    if v >= 2000:
        return f"회전수 {v:.0f}rpm으로 평균적인 수준"
    return f"회전수 {v:.0f}rpm으로 낮은 편(스플리터 등 저회전 구종에서 흔함)"


def _fallback_explanation(input_dict, pred_label, confidence, proba_dict, p_throws):
    """Gemini 호출이 불가능하거나 실패했을 때 사용하는 규칙 기반 설명."""
    name = PITCH_NAMES.get(pred_label, pred_label)
    ax_desc = _ax_direction_desc(input_dict['ax'], p_throws)
    reasons = [
        _describe_speed(input_dict['release_speed']),
        _describe_az(input_dict['az']),
        _describe_spin(input_dict['release_spin_rate']),
        f"수평 움직임은 {ax_desc} 궤적",
    ]

    sentence = (
        f"{', '.join(reasons)} 등의 특징을 종합하여 모델은 이 투구를 "
        f"{pred_label}({name})로 분류했습니다 (신뢰도 {confidence * 100:.1f}%)."
    )

    runner_up_label, runner_up_conf = _runner_up(pred_label, proba_dict)
    if runner_up_conf >= 0.15:
        runner_up_name = PITCH_NAMES.get(runner_up_label, runner_up_label)
        sentence += (
            f" 다음으로 가능성이 높았던 구종은 {runner_up_label}({runner_up_name})로 "
            f"{runner_up_conf * 100:.1f}%의 확률을 보였습니다."
        )

    return sentence


def generate_explanation(input_dict, pred_label, confidence, proba_dict, p_throws='R'):
    """Gemini API로 자연어 설명을 생성한다. 키가 없거나 호출에 실패하면 규칙 기반으로 폴백한다."""
    if _model is not None:
        try:
            prompt = _build_prompt(input_dict, pred_label, confidence, proba_dict, p_throws)
            response = _model.generate_content(
                prompt,
                generation_config={"temperature": 1.0},
                request_options={"timeout": 15},
            )
            text = response.text.strip()
            if text:
                return text
        except Exception:
            pass

    return _fallback_explanation(input_dict, pred_label, confidence, proba_dict, p_throws)
