# model.py - MLB 구종 분류 MLP 모델 정의 (TensorFlow/Keras)

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'pitch_model.h5')

os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(input_dim, num_classes, learning_rate=0.001):
    """7개 구종을 분류하는 3층 MLP 모델을 생성하고 컴파일한다."""
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),

        # 은닉층 1
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # 은닉층 2
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # 은닉층 3
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),

        # 출력층 - 구종 수만큼 소프트맥스
        layers.Dense(num_classes, activation='softmax'),

    ], name='MLB_Pitch_MLP')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_callbacks(patience=10):
    """EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 콜백 목록을 반환한다."""
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]


def load_model():
    """저장된 .h5 모델 파일을 불러온다."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"모델 파일 없음: {MODEL_PATH}\n train.py를 먼저 실행하세요."
        )
    model = keras.models.load_model(MODEL_PATH)
    print(f"[모델] 불러오기 완료: {MODEL_PATH}")
    return model


if __name__ == '__main__':
    model = build_model(input_dim=17, num_classes=7)
    model.summary()
