import numpy as np
import struct
import gzip
import os
import matplotlib.pyplot as plt

# Fashion-MNIST 데이터 로드 함수
def load_fashion_mnist(path, kind='train'):
    """Fashion-MNIST 데이터를 로드하는 함수"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
        images = images.astype(np.float32) / 255.0  # 정규화
    
    return images, labels

# Adam 옵티마이저 클래스
class Adam:
    """Adam Optimizer"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            # m and v 업데이트
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            # 파라미터 업데이트
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

# 신경망 클래스 정의
class NeuralNetwork:
    def __init__(self, layers, optimizer=None, dropout_ratio=0.5, use_dropout=False):
        """신경망 초기화"""
        self.layers = layers
        self.params = {}
        self.cache = {}
        self.optimizer = optimizer
        self.dropout_ratio = dropout_ratio  # 드롭아웃 비율
        self.use_dropout = use_dropout  # 드롭아웃을 사용할지 여부
        self.initialize_weights()
    
    def initialize_weights(self):
        """가중치와 편향 초기화"""
        for i in range(1, len(self.layers)):
            self.params['W' + str(i)] = np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2. / self.layers[i-1])
            self.params['b' + str(i)] = np.zeros((1, self.layers[i]))
    
    def relu(self, x):
        """ReLU 활성화 함수"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU 도함수"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax 함수"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """순전파"""
        A = X
        self.cache['A0'] = A
        for i in range(1, len(self.layers)):
            Z = np.dot(A, self.params['W' + str(i)]) + self.params['b' + str(i)]
            self.cache['Z' + str(i)] = Z
            if i == len(self.layers) - 1:
                A = self.softmax(Z)
            else:
                A = self.relu(Z)
                # 드롭아웃 적용
                if self.use_dropout:
                    dropout_mask = np.random.rand(*A.shape) > self.dropout_ratio
                    A *= dropout_mask
                    self.cache['dropout_mask' + str(i)] = dropout_mask
            self.cache['A' + str(i)] = A
        return A
    
    def compute_loss(self, Y_pred, Y_true):
        """교차 엔트로피 손실 함수"""
        m = Y_true.shape[0]
        log_likelihood = -np.log(Y_pred[range(m), Y_true])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, X, Y_true):
        """역전파"""
        m = X.shape[0]
        grads = {}
        Y_pred = self.cache['A' + str(len(self.layers)-1)]
        dZ = Y_pred
        dZ[range(m), Y_true] -= 1
        dZ /= m
        
        for i in reversed(range(1, len(self.layers))):
            grads['W' + str(i)] = np.dot(self.cache['A' + str(i-1)].T, dZ)
            grads['b' + str(i)] = np.sum(dZ, axis=0, keepdims=True)
            if i > 1:
                dA_prev = np.dot(dZ, self.params['W' + str(i)].T)
                dZ = dA_prev * self.relu_derivative(self.cache['Z' + str(i-1)])
                # 드롭아웃 적용
                if self.use_dropout:
                    dZ *= self.cache['dropout_mask' + str(i-1)]
        
        return grads
    
    def update_params(self, grads):
        """가중치와 편향 업데이트 (Adam 옵티마이저 사용)"""
        if self.optimizer:
            self.optimizer.update(self.params, grads)
    
    def predict(self, X):
        """예측"""
        Y_pred = self.forward(X)
        return np.argmax(Y_pred, axis=1)

# 학습 함수
def train(model, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=64):
    """모델 학습 함수"""
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        Y_shuffled = Y_train[permutation]
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            Y_pred = model.forward(X_batch)
            loss = model.compute_loss(Y_pred, Y_batch)
            grads = model.backward(X_batch, Y_batch)
            model.update_params(grads)
        
        # 에포크 끝난 후 전체 데이터에 대해 평가
        Y_train_pred = model.forward(X_train)
        train_loss = model.compute_loss(Y_train_pred, Y_train)

        train_acc = np.mean(np.argmax(Y_train_pred, axis=1) == Y_train)
        
        Y_val_pred = model.forward(X_val)
        val_loss = model.compute_loss(Y_val_pred, Y_val)
        val_acc = np.mean(np.argmax(Y_val_pred, axis=1) == Y_val)
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}')
    
    return history

# 메인 실행
if __name__ == "__main__":
    # 데이터 로드
    X_train, Y_train = load_fashion_mnist('dataset', kind='train')
    X_test, Y_test = load_fashion_mnist('dataset', kind='t10k')
    
    # 검증 데이터 분리
    validation_size = 5000
    X_val = X_train[:validation_size]
    Y_val = Y_train[:validation_size]
    X_train = X_train[validation_size:]
    Y_train = Y_train[validation_size:]
    
    # Adam 옵티마이저 설정
    adam_optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
    
    # 신경망 모델 설정 
    layers = [784, 128, 64, 10]
    model = NeuralNetwork(layers, optimizer=adam_optimizer, dropout_ratio=0.5, use_dropout=True)
    
    # 모델 학습
    history = train(model, X_train, Y_train, X_val, Y_val, epochs=20, batch_size=64)
    
    # 테스트 데이터 평가
    Y_test_pred = model.predict(X_test)
    test_acc = np.mean(Y_test_pred == Y_test)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # 학습 과정 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Function')

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()
