"""explain.py - 예측 결과에 대한 규칙 기반 자연어 설명 생성"""

PITCH_NAMES = {
    'FF': '포심 패스트볼', 'SI': '싱커',    'SL': '슬라이더',
    'CU': '커브',          'CH': '체인지업', 'FC': '커터', 'FS': '스플리터',
}


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


def generate_explanation(input_dict, pred_label, confidence, proba_dict):
    """규칙 기반으로 예측 근거를 한국어 문장으로 조합한다."""
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

    runner_up = sorted(
        ((label, p) for label, p in proba_dict.items() if label != pred_label),
        key=lambda item: item[1], reverse=True,
    )[0]
    if runner_up[1] >= 0.15:
        r_label, r_conf = runner_up
        r_name = PITCH_NAMES.get(r_label, r_label)
        sentence += (
            f" 다음으로 가능성이 높았던 구종은 {r_label}({r_name})로 "
            f"{r_conf * 100:.1f}%의 확률을 보였습니다."
        )

    return sentence
