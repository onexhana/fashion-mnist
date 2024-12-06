import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.utils import to_categorical

# 데이터 로드
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 옵티마이저 종류 리스트
optimizers = {
    'SGD': SGD(),
    'Adam': Adam(),
    'RMSprop': RMSprop(),
    'Adagrad': Adagrad()
}

# 모델 학습 함수 정의
def create_model(optimizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# 각 옵티마이저에 대해 학습 진행
history_dict = {}
for name, optimizer in optimizers.items():
    print(f"Training with {name} optimizer...")
    model = create_model(optimizer)
    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=10,
                        batch_size=32,
                        verbose=0)  # 학습 과정을 표시하지 않음
    history_dict[name] = history.history

# 학습 곡선 시각화
plt.figure(figsize=(10, 6))
for name, history in history_dict.items():
    plt.plot(history['accuracy'], label=f'{name} - Train Accuracy')
    plt.plot(history['val_accuracy'], label=f'{name} - Validation Accuracy')

plt.title('Comparison of Optimizers')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 평가: 각 옵티마이저에 대해 테스트 정확도 출력
for name, optimizer in optimizers.items():
    model = create_model(optimizer)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy with {name} optimizer: {test_acc:.4f}")
