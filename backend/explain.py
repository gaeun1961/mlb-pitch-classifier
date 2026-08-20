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
    _model = genai.GenerativeModel('gemini-3.6-flash')


def _runner_up(pred_label, proba_dict):
    return sorted(
        ((label, p) for label, p in proba_dict.items() if label != pred_label),
        key=lambda item: item[1], reverse=True,
    )[0]


def _build_prompt(input_dict, pred_label, confidence, proba_dict):
    name = PITCH_NAMES.get(pred_label, pred_label)
    runner_up_label, runner_up_conf = _runner_up(pred_label, proba_dict)
    runner_up_name = PITCH_NAMES.get(runner_up_label, runner_up_label)

    return f"""당신은 MLB 투구 데이터 분석 전문가입니다. 아래 투구가 왜 이 구종으로 분류되었는지
2~3문장의 자연스러운 한국어로 설명해주세요. 피처 이름을 나열하지 말고 야구 팬이 이해하기 쉽게 풀어써주세요.

예측 구종: {pred_label} ({name}), 신뢰도 {confidence * 100:.1f}%
2순위 구종: {runner_up_label} ({runner_up_name}), 확률 {runner_up_conf * 100:.1f}%

핵심 피처:
- 구속(release_speed): {input_dict['release_speed']:.1f} mph
- 수직 가속도(az, 양수일수록 덜 떨어지고 음수일수록 급격히 가라앉음): {input_dict['az']:.1f}
- 수평 가속도(ax, 좌우로 휘는 정도): {input_dict['ax']:.1f}
- 회전수(release_spin_rate): {input_dict['release_spin_rate']:.0f} rpm

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


def _fallback_explanation(input_dict, pred_label, confidence, proba_dict):
    """Gemini 호출이 불가능하거나 실패했을 때 사용하는 규칙 기반 설명."""
    name = PITCH_NAMES.get(pred_label, pred_label)
    reasons = [
        _describe_speed(input_dict['release_speed']),
        _describe_az(input_dict['az']),
        _describe_spin(input_dict['release_spin_rate']),
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


def generate_explanation(input_dict, pred_label, confidence, proba_dict):
    """Gemini API로 자연어 설명을 생성한다. 키가 없거나 호출에 실패하면 규칙 기반으로 폴백한다."""
    if _model is not None:
        try:
            prompt = _build_prompt(input_dict, pred_label, confidence, proba_dict)
            response = _model.generate_content(
                prompt,
                request_options={"timeout": 10},
            )
            text = response.text.strip()
            if text:
                return text
        except Exception:
            pass

    return _fallback_explanation(input_dict, pred_label, confidence, proba_dict)
