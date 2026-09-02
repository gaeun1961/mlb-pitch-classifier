# train.py - 모델 학습 및 결과 저장

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score

from data_loader import load_data, FEATURE_COLS
from model import build_model, get_callbacks

matplotlib.use('Agg')  # GUI 없는 환경에서 그래프 저장

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_loss_accuracy(history):
    """학습 곡선(loss, accuracy)을 그래프로 저장한다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 그래프
    ax1.plot(history.history['loss'],     label='train')
    ax1.plot(history.history['val_loss'], label='val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy 그래프
    ax2.plot(history.history['accuracy'],     label='train')
    ax2.plot(history.history['val_accuracy'], label='val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    path = os.path.join(RESULTS_DIR, 'loss_accuracy.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f'[결과] 학습 곡선 저장: {path}')


def save_confusion_matrix(y_test, y_pred, label_map):
    """테스트 세트에 대한 혼동 행렬을 계산하고 이미지로 저장한다."""
    labels = [label_map[i] for i in range(len(label_map))]

    cm  = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix')

    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f'[결과] 혼동 행렬 저장: {path}')


def save_metrics_summary(y_test, y_pred, label_map, acc):
    """정확도·weighted F1·구종별 classification report를 PR/기록용 마크다운으로 저장한다."""
    labels = [label_map[i] for i in range(len(label_map))]
    report = classification_report(y_test, y_pred, target_names=labels, digits=2)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    path = os.path.join(RESULTS_DIR, 'metrics.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"**테스트 정확도**: {acc * 100:.2f}%  \n")
        f.write(f"**Weighted F1**: {weighted_f1:.4f}  \n")
        f.write(f"**테스트 세트**: {len(y_test):,}건\n\n")
        f.write("```\n" + report + "\n```\n")

    print(f'[결과] 메트릭 요약 저장: {path}')
    return weighted_f1


def train():
    """데이터 로드부터 학습, 평가, 결과 저장까지 전체 파이프라인을 실행한다."""
    X_train, X_val, X_test, y_train, y_val, y_test, label_map = load_data()

    input_dim   = X_train.shape[1]
    num_classes = len(label_map)

    model = build_model(input_dim, num_classes)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=512,
        callbacks=get_callbacks(patience=10),
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\n[평가] 테스트 정확도: {acc * 100:.2f}%')
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    save_loss_accuracy(history)
    save_confusion_matrix(y_test, y_pred, label_map)
    save_metrics_summary(y_test, y_pred, label_map, acc)


if __name__ == '__main__':
    train()
