import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd

class Network(object):
    def __init__(self, learning_rate = 0.01, epoches = 500):
              
        # set the size of layer
        self.ip_size = 34
        self.h1_size = 24 #24
        self.h2_size = 16 #16
        self.op_size = 2

        #init the weight(高斯分佈初始化)
        self.w1 = np.random.normal(size = (self.ip_size, self.h1_size)) #34*10
        self.w2 = np.random.normal(size = (self.h1_size, self.h2_size))
        self.w3 = np.random.normal(size = (self.h2_size, self.op_size))

        #other class variable
        self.Loss_list = []
        self.x_train = []
        self.y_train = []
        self.learning_rate = learning_rate
        self.epoches = epoches
        
    def init_bias(self):
        self.b1 = np.random.randn(1, self.h1_size)
        self.b2 = np.random.randn(1, self.h2_size)
        self.b3 = np.random.randn(1, self.op_size)

    def get_data(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.batchsize = self.x_train.shape[0]
        print("train data shape: {} , {}".format(np.shape(x_train), np.shape(y_train)))
        self.init_bias()


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative_sigmoid(self, x): # sigmoid(x) 的微分 = sigmoid(x) * (1 - sigmoid(x))
        return np.multiply(x, 1-x)
    

    def forward(self, x, input_size, no_softmax):
        self.x1 = np.dot(x, self.w1) + np.tile(self.b1, (input_size, 1))
        self.a1 = self.sigmoid(self.x1)
        self.x2 = np.dot(self.a1, self.w2) + np.tile(self.b2, (input_size, 1))
        self.a2 = self.sigmoid(self.x2)
        self.x3 = np.dot(self.a2, self.w3) + np.tile(self.b3, (input_size, 1))
        # self.a3 = self.sigmoid(self.x3)        
        
        if not no_softmax:
            self.y_pred = np.apply_along_axis(self.softmax, 1, self.x3) # 280*2
        else:
            self.y_pred = self.x3

        return self.y_pred

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()
    
    def softmax_derivative(self, x):
        s = self.softmax(x)
        d_softmax = np.diag(s) - np.outer(s, s)
        return d_softmax
    
    def categorical_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))
    
    def categorical_cross_entropy_derivative(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred # 1*2
    
    def back_propogation(self):
        # loss calculation   
        entropy_loss = 0
        for i in range(len(self.y_pred)):
            entropy_loss += self.categorical_cross_entropy(self.y_train[i], self.y_pred[i])
        self.Loss_list.append(entropy_loss)
        
        w3_forward = self.y_pred - self.y_train # cross-entropy加上softmax的backpropagation結果: z對loss微分
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
            self.forward(self.x_train, self.batchsize, no_softmax=False)
            self.back_propogation()
            if (i + 1) % 50 == 0:
                print("epoch {} loss : {}".format(i+1, self.Loss_list[i]))
        plt.plot(self.Loss_list)
        plt.title("training curve with lr={}".format(self.learning_rate))
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()
    
    def calculate_error(self):
        # Train results
        self.train_pred = self.forward(self.x_train, self.x_train.shape[0], no_softmax=True)
        pred_good_index = np.where(self.train_pred[:,0] >= self.train_pred[:,1])
        pred_bad_index = np.where(self.train_pred[:,0] < self.train_pred[:,1])
        gt_good_index = np.where(self.y_train[:,0] == 1)[0]
        gt_bad_index = np.where(self.y_train[:,0] == 0)[0]
        
        error_num = np.shape(np.where(self.y_train[pred_good_index,0] == 0)[1])[0] + np.shape(np.where(self.y_train[pred_bad_index,0] == 1)[1])[0]
        error_rate = error_num / len(self.train_pred)
        print("Train Error Rate: {}".format(error_rate))
        
        plt.plot(self.train_pred[gt_good_index,0], self.train_pred[gt_good_index,1], linestyle='None', marker='o', markersize=4, color='blue', label='good') #good
        plt.plot(self.train_pred[gt_bad_index,0], self.train_pred[gt_bad_index,1], linestyle='None', marker='o', markersize=4, color='red', label='bad') #bad
        plt.legend(loc='upper right')
        plt.title("Train 2D feature {} epoch".format(self.epoches))
        plt.ylabel("y2")
        plt.xlabel("y1")
        plt.show()
        
        # Test results
        self.test_pred = self.forward(self.x_test, self.x_test.shape[0], no_softmax=True)
        pred_good_index = np.where(self.test_pred[:,0] >= self.test_pred[:,1])
        pred_bad_index = np.where(self.test_pred[:,0] < self.test_pred[:,1])
        gt_good_index = np.where(self.y_test[:,0] == 1)[0]
        gt_bad_index = np.where(self.y_test[:,0] == 0)[0]
        
        error_num = np.shape(np.where(self.y_test[pred_good_index,0] == 0)[1])[0] + np.shape(np.where(self.y_test[pred_bad_index,0] == 1)[1])[0]
        error_rate = error_num / len(self.test_pred)
        print("Test Error Rate: {}".format(error_rate))
        
        plt.plot(self.test_pred[gt_good_index,0], self.test_pred[gt_good_index,1], linestyle='None', marker='o', markersize=4, color='blue', label='good') #good
        plt.plot(self.test_pred[gt_bad_index,0], self.test_pred[gt_bad_index,1], linestyle='None', marker='o', markersize=4, color='red', label='bad') #bad
        plt.legend(loc='upper right')
        plt.title("Test 2D feature {} epoch".format(self.epoches))
        plt.ylabel("y2")
        plt.xlabel("y1")
        plt.show()

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

if __name__ == '__main__':
    
    # read datas
    df = pd.read_csv('ionosphere_data.csv')
    data_list = df.values.tolist()
    
    # g,b to 1,0
    for i in range(len(data_list)):
        if data_list[i][-1] == 'g':
            data_list[i][-1] = [1, 0]
        elif data_list[i][-1] == 'b':
            data_list[i][-1] = [0, 1]
    data_array = down_dimention(data_list)
    
    # shuffle
    np.random.shuffle(data_array)
    
    # categorize datas
    train_input = data_array[:int(0.80*len(data_array)), :-2] # N組*輸入數量34
    train_output = data_array[:int(0.80*len(data_array)), -2:] # N組*輸出數量2
    test_inout = data_array[int(0.80*len(data_array)):, :-2]
    test_output = data_array[int(0.80*len(data_array)):, -2:]
    
    # train
    demo = Network(learning_rate=1e-5, epoches=390)
    # demo = Network(learning_rate=1e-5, epoches=10)
    # demo = Network(learning_rate=1e-5, epoches=2000)
    demo.get_data(train_input, train_output, test_inout, test_output)
    demo.train()
    demo.calculate_error()