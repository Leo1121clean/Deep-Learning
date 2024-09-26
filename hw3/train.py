import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint

# 自訂回調函數，用於保存模型
class SaveModelCallback(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', save_freq='epoch', num_save_points=1, **kwargs):
        super().__init__(filepath, monitor=monitor, save_freq=save_freq, save_best_only=False, **kwargs) # save_best_only:只有validation比train改善才會儲存權重
        self.num_save_points = num_save_points

    def on_epoch_end(self, epoch, logs=None):
        # if (epoch + 1) % self.num_save_points == 0:
        # print(f"Saving model for epoch {epoch + 1}")
        super().on_epoch_end(epoch, logs)

if __name__ == '__main__':
    # Preprocessing of Text
    train_data_url = 'shakespeare_train.txt'
    val_data_url = 'shakespeare_valid.txt'

    with io.open(train_data_url, 'r', encoding='utf-8') as f:
        train_text = f.read()

    with io.open(val_data_url, 'r', encoding='utf-8') as f:
        val_text = f.read()

    # 創建詞彙表
    vocab = sorted(set(train_text + val_text))
    vocab_size = len(vocab)

    # 創建字符到整數的映射和反映射
    char_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_char = {i: c for i, c in enumerate(vocab)}

    # 將文本轉換為整數序列
    train_int_sequence = [char_to_int[char] for char in train_text]
    val_int_sequence = [char_to_int[char] for char in val_text]

    # 定義輸入序列的長度
    input_length = 50

    # 創建訓練集和驗證集的輸入序列和目標序列
    def create_sequences(int_sequence):
        input_sequences = []
        target_sequences = []

        for i in range(len(int_sequence) - input_length):
            input_seq = int_sequence[i:i + input_length]
            target_seq = int_sequence[i + input_length]
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

        return np.array(input_sequences), np.array(target_sequences)

    # 創建訓練集和驗證集的輸入序列和目標序列
    X_train, y_train = create_sequences(train_int_sequence)
    X_val, y_val = create_sequences(val_int_sequence)

    # 將輸入序列進行填充（如果需要）
    X_train_padded = pad_sequences(X_train, maxlen=input_length, padding='pre')
    X_val_padded = pad_sequences(X_val, maxlen=input_length, padding='pre')

    # 定義模型
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=256, input_length=input_length),
        # SimpleRNN(64),
        LSTM(256),
        Dense(vocab_size, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    
    # 使用自訂的 SaveModelCallback 回調保存模型
    save_model_callback = SaveModelCallback(
        'weight/shakespeare_rnn_model_epoch_{epoch}.h5', 
        save_freq=1, # 保存每個 epoch
        num_save_points=1  # 設定保存模型的間隔
    )

    history = model.fit(X_train_padded, y_train, epochs=40, batch_size=64, validation_data=(X_val_padded, y_val), callbacks=[save_model_callback])
    # history = model.fit(X_train_padded, y_train, epochs=40, batch_size=64, validation_data=(X_val_padded, y_val))

    result = {'name': 'RNN_MODEL', 'history': history, 'model': model}
    
    plt.plot(result['history'].history['loss'], label='Sparse Categorical Cross Entrophy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()
    
    plt.plot(result['history'].history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(result['history'].history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')
    plt.show()