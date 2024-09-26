import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd

class Network(object):
    def __init__(self, learning_rate = 0.01, epoches = 500):
              
        # set the size of layer
        self.h1_size = 10
        self.h2_size = 10
        self.op_size = 1
        
        # other class variable
        self.Loss_list = []
        self.x_train = []
        self.y_train = []
        self.learning_rate = learning_rate
        self.epoches = epoches
        
    def init_param(self):
        self.ip_size = np.shape(self.x_train[1])[0] # 根據輸入數量給size
        
        #init the weight(高斯分佈初始化)
        self.w1 = np.random.normal(size = (self.ip_size, self.h1_size))
        self.w2 = np.random.normal(size = (self.h1_size, self.h2_size))
        self.w3 = np.random.normal(size = (self.h2_size, self.op_size))
        
        self.b1 = np.random.randn(1, self.h1_size) #1*10
        self.b2 = np.random.randn(1, self.h2_size)
        self.b3 = np.random.randn(1, self.op_size)

    def get_data(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.batchsize = self.x_train.shape[0]
        print("train data shape: {} , {}".format(np.shape(x_train), np.shape(y_train)))
        self.init_param()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative_sigmoid(self, x): # sigmoid(x) 的微分 = sigmoid(x) * (1 - sigmoid(x))
        return np.multiply(x, 1-x)
    

    def forward(self, x, input_size):
        self.x1 = np.dot(x, self.w1) + np.tile(self.b1, (input_size, 1)) # 576*10 + 576*10
        self.a1 = self.sigmoid(self.x1)
        self.x2 = np.dot(self.a1, self.w2) + np.tile(self.b2, (input_size, 1)) # 576*10 + 576*10
        self.a2 = self.sigmoid(self.x2)
        self.x3 = np.dot(self.a2, self.w3) + np.tile(self.b3, (input_size, 1)) # 576*1 + 576*1
        # self.y_pred = self.sigmoid(self.x3)
        self.y_pred = self.x3

        return self.y_pred # 576*1
    
    def RMS_Loss(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def RMS_prime(self, y_true, y_pred): # 用y_pred微
        return -(y_true - y_pred) / (len(y_true) * np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def SOS_Loss(self, y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2)
    
    def SOS_prime(self, y_true, y_pred):
        # return np.sum(-2 * (y_true - y_pred))
        return -2 * (y_true - y_pred)
    
    def back_propogation(self):
        # calculate loss
        self.Loss_list.append(self.SOS_Loss(self.y_train, self.y_pred))

        w3_forward = self.SOS_prime(self.y_train, self.y_pred) # 576*1   
        w3_grad = np.dot(self.a2.T, w3_forward)
        w2_forward = w3_forward.dot(self.w3.T)*self.derivative_sigmoid(self.a2)
        w2_grad =  np.dot(self.a1.T, w2_forward)
        w1_forward = np.dot(w2_forward,self.w2.T)*self.derivative_sigmoid(self.a1)
        w1_grad = np.dot(self.x_train.T,w1_forward)
        
        # update weight for "batchsize" times
        for batchsize_index in range(self.batchsize):
            self.w3 -= self.learning_rate * w3_grad
            self.w2 -= self.learning_rate * w2_grad
            self.w1 -= self.learning_rate * w1_grad
            self.b3 -= self.learning_rate * w3_forward[batchsize_index][0]
            self.b2 -= self.learning_rate * w2_forward[batchsize_index][0]
            self.b1 -= self.learning_rate * w1_forward[batchsize_index][0]
    
    def train(self):
        for i in range(self.epoches):
            self.forward(self.x_train, self.batchsize)
            self.back_propogation()
            if (i + 1) % 1000 == 0:
                print("epoch {} loss : {}".format(i+1, self.Loss_list[i]))
                if i == self.epoches -1:
                    self.y_final = copy.deepcopy(self.y_pred)
        plt.plot(self.Loss_list)
        plt.title("training curve with lr={}".format(self.learning_rate))
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()
    
    def calculate_error(self):
        
        self.train_pred = self.forward(self.x_train, self.x_train.shape[0])
        self.train_error = self.RMS_Loss(self.y_train, self.train_pred)
        print("Train RMS Error: {}".format(self.train_error))
        
        self.test_pred = self.forward(self.x_test, self.x_test.shape[0])
        self.test_error = self.RMS_Loss(self.y_test, self.test_pred)
        print("Test RMS Error: {}".format(self.test_error))
        
        # print("Train Error Sum: {}".format(np.mean(abs(self.y_train - self.train_pred))))
        # print("Test Error Sum: {}".format(np.mean(abs(self.y_test - self.test_pred))))
        
        # Train results
        plt.plot(self.train_pred, label='predict')
        plt.plot(self.y_train, label='label')
        plt.title("prediction for training data")
        plt.ylabel("Heating Load")
        plt.xlabel("#th case")
        plt.legend(loc='upper right')
        plt.show()
        
        # Test results
        plt.plot(self.test_pred, label='predict')
        plt.plot(self.y_test, label='label')
        plt.title("prediction for test data")
        plt.ylabel("Heating Load")
        plt.xlabel("#th case")
        plt.legend(loc='upper right')
        plt.show()
        


def create_one_hot_encoding(data_list, column_index):
    column_list = [int(row[column_index]) for row in data_list]
    one_hot_encoded = np.eye(max(column_list) - min(column_list) + 1)
    
    for i in range(len(data_list)):
        data_list[i][column_index] = one_hot_encoded[int(data_list[i][column_index] - min(column_list))].tolist()

def down_dimention(data_list):
    final_list = []
    
    for nested_list in data_list:
        flat_list = []
        
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)
        final_list.append(flat_list)
    
    final_array = np.array(final_list)
    return final_array

def normalize(data_list, index):
    column_to_normalize = [row[index] for row in data_list]
    min_value = min(column_to_normalize)
    max_value = max(column_to_normalize)
    for row in data_list:
        row[index] = (row[index] - min_value) / (max_value - min_value)

if __name__ == '__main__':
    
    # read datas
    df = pd.read_csv('energy_efficiency_data.csv')
    data_list = df.values.tolist()
    
    # one hot encoding
    create_one_hot_encoding(data_list, column_index=5) # orientation
    create_one_hot_encoding(data_list, column_index=7) # glazing area distribution
    data_array = down_dimention(data_list) # 將one hot的list展開成一維
    
    # shuffle
    np.random.shuffle(data_array)
    
    # normalize
    normalize(data_array, 1)
    normalize(data_array, 2)
    normalize(data_array, 3)
    
    # remove specific datas
    columns_to_remove = []
    data_array = np.delete(data_array, columns_to_remove, axis=1)

    # categorize datas
    train_input = data_array[:int(0.75*len(data_array)), :-2] # N組*輸入數量16
    train_output = data_array[:int(0.75*len(data_array)), -2:-1] # N組*輸出數量1
    test_inout = data_array[int(0.75*len(data_array)):, :-2]
    test_output = data_array[int(0.75*len(data_array)):, -2:-1]
    
    # train
    demo = Network(learning_rate=1e-8, epoches=3000)
    demo.get_data(train_input, train_output, test_inout, test_output)
    demo.train()
    demo.calculate_error()