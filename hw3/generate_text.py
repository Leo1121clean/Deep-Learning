import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_text(model, prime_text, char_to_int, int_to_char, length=200):
    generated_text = prime_text

    for _ in range(length):
        # 將起始文本轉換為整數序列
        input_sequence = [char_to_int[char] for char in prime_text]

        # 將序列填充到所需的輸入長度
        input_sequence = pad_sequences([input_sequence], maxlen=model.input_shape[1], padding='pre')

        # 預測下一個字符
        predicted_probs = model.predict(input_sequence)[0]

        # 根據預測的概率選擇下一個字符
        next_char_index = np.random.choice(len(char_to_int), p=predicted_probs)

        # 將索引轉換回字符
        next_char = int_to_char[next_char_index]

        # 將下一個字符添加到生成的文本中
        generated_text += next_char

        # 更新起始文本以進行下一次迭代
        prime_text = prime_text[1:] + next_char

    return generated_text

if __name__ == '__main__':

    # 讀取訓練集文本
    with io.open('shakespeare_train.txt', 'r', encoding='utf-8') as f:
        train_text = f.read()

    # 讀取驗證集文本
    with io.open('shakespeare_valid.txt', 'r', encoding='utf-8') as f:
        val_text = f.read()

    # 創建詞彙表
    vocab = sorted(set(train_text + val_text))

    # 創建字符到整數的映射和反映射
    char_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_char = {i: c for i, c in enumerate(vocab)}

    # 將文本轉換為整數序列
    train_int_sequence = [char_to_int[char] for char in train_text]
    val_int_sequence = [char_to_int[char] for char in val_text]

    # 定義輸入序列的長度
    input_length = 10

    # 創建輸入序列和目標序列
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

    # 讀取已經訓練好的模型
    path = 'weight/RNN256_seq10/'
    model = load_model(path + 'shakespeare_rnn_model_epoch_40.h5')

    # # 計算訓練集誤差率
    # train_loss = model.evaluate(X_train, y_train, verbose=0)
    # print(f'Training Error Rate: {train_loss}')

    # # 計算驗證集誤差率
    # val_loss = model.evaluate(X_val, y_val, verbose=0)
    # print(f'Validation Error Rate: {val_loss}')
    
    ######################### input prime text #########################
    # 用於生成輸出的起始文本
    prime_text = 'JULIET'
    # prime_text = 'We are accounted poor citizens, the patricians good.'

    # 生成文本
    generated_text = generate_text(model, prime_text, char_to_int, int_to_char, length=300)

    # 打印生成的文本
    print(generated_text)