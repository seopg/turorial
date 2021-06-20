# 뇌를 자극하는 파이썬 예제
# https://www.hanbit.co.kr/media/community/brain_default_board_view.html?bd_id=python&board_idx=19250&page=0

# 자바와 파이썬의 차이
# https://m.blog.naver.com/PostView.nhn?blogId=magnking&logNo=220907439135&proxyReferer=https:%2F%2Fwww.google.com%2F

# 파이썬코딩 한시간만에 배우기
# https://www.youtube.com/watch?v=M6kQTpIqpLs

# numpy 예제
# http://pythonstudy.xyz/python/article/402-numpy-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0

# pandas 예제
# http://pythonstudy.xyz/python/article/408-pandas-%EB%8D%B0%EC%9D%B4%ED%83%80-%EB%B6%84%EC%84%9D

# dataframe 예제
# http://pythonstudy.xyz/python/article/408-pandas-%EB%8D%B0%EC%9D%B4%ED%83%80-%EB%B6%84%EC%84%9D

# 케라스를 활용한 간단한 예제 살펴보기
# https://cyc1am3n.github.io/2018/11/02/introduction-to-keras.html

# 케라스 김태영 T아카데미 1,2강
# https://www.youtube.com/watch?v=cJpjAmRO_h8
# https://www.youtube.com/watch?v=IPR2bYFa6Rw
# https://www.youtube.com/watch?v=QldOyLAQfVg

# 케라스 김태영 블로그
# https://tykimos.github.io/lecture/

# 코딩셰프 케라스맛
# https://github.com/jskDr/keraspp.git

# 케라스 창시자에게서 배우는 딥러닝
# https://github.com/gilbutITbook/006975

# 코랩 소개
# https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=C4HZx7Gndbrh

# 코랩 끊김 방지
# https://teddylee777.github.io/colab/google-colab-%EB%9F%B0%ED%83%80%EC%9E%84-%EC%97%B0%EA%B2%B0%EB%81%8A%EA%B9%80%EB%B0%A9%EC%A7%80

# 이미지 분류
# https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb#scrollTo=dlauq-4FWGZM






###########################################################################################################################################################

# 파이썬 예제
print("Hello, World.")



###########################################################################################################################################################

# 입력받아 연산
print("첫 번째 수를 입력하세요. : ")
a = input()
print("두 번째 수를 입력하세요. : ")
b = input()

result = int(a) * int(b)

print("{0} * {1} = {2}".format(a, b, result))


###########################################################################################################################################################
# 파이썬 예제
for s in ['뇌를', '자극하는', '파이썬'] :
    print(s)



###########################################################################################################################################################
# 파이썬 예제
for s in '뇌를 자극하는 파이썬' :
    print(s)



###########################################################################################################################################################
# 파이썬 예제 for
for i in (1, 2, 3) :
	print(i)



###########################################################################################################################################################
# 파이썬 예제 for range
for i in range(1, 6) :
        for j in range(i) :
            print( "*", end = "", )
        
        print()


###########################################################################################################################################################
import keras
keras.__version__

###########################################################################################################################################################
# 쥬피터에서
# ipynb 를 import 하기

!pip install import_ipynb
import import_ipynb
!pwd
%cd "/content/drive/My Drive/Colab Notebooks"



import calculator

print(calculator.plus(10, 5))
print(calculator.minus(10, 5))
print(calculator.multiply(10, 5))
print(calculator.divide(10, 5))



###########################################################################################################################################################
# numpy 예제

import numpy as np
 
list1 = [1, 2, 3, 4]
a = np.array(list1)
print(a.shape) # (4, )
 
b = np.array([[1,2,3],[4,5,6]])
print(b.shape) # (2, 3)
print(b[0,0])  # 1  



###########################################################################################################################################################
import numpy as np
 
a = np.zeros((2,2))
print(a)
# 출력:
# [[ 0.  0.]
#  [ 0.  0.]]
 
a = np.ones((2,3))
print(a)
# 출력:
# [[ 1.  1.  1.]
#  [ 1.  1.  1.]]
 
a = np.full((2,3), 5)
print(a)
# 출력:
# [[5 5 5]
#  [5 5 5]]
 
a = np.eye(3)
print(a)
# 출력:
# [[ 1.  0.  0.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]]
 
a = np.array(range(20)).reshape((4,5))
print(a)
# 출력:
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]]



###########################################################################################################################################################
# slice 예제
a = ['a', 'b', 'c', 'd', 'e']

print("a[ 1 :  ] = ",a[ 1 :  ])
print("a[  : 2 ] = ",a[  : 2 ])
print("a[ 2 : 4 ] = ",a[ 2 : 4 ])
print("a[ : : 2 ] = ", a[ : : 2 ])


###########################################################################################################################################################
# pandas 예제
import pandas as pd
 
data = {
    'year': [2016, 2017, 2018],
    'GDP rate': [2.8, 3.1, 3.0],
    'GDP': ['1.637M', '1.73M', '1.83M']
}
 
df = pd.DataFrame(data)

print(df)


###########################################################################################################################################################
# 케라스 창시자에게서 배우는 딥러닝
! git clone https://github.com/gilbutITbook/006975.git


###########################################################################################################################################################
# 코딩셰프 케라스맛
! git clone https://github.com/jskDr/keraspp.git


###########################################################################################################################################################
# 코딩셰프 케라스맛
import keras
import numpy

x = numpy.array([0, 1, 2, 3, 4])
y = x * 2 + 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x[:2], y[:2], epochs=1000, verbose=0)

print('Targets:', y[2:])
print('Predictions:', model.predict(x[2:]).flatten())



###########################################################################################################################################################
# 코딩셰프 케라스맛 응용
import keras
import numpy

x = numpy.array([(1,1), (1,2), (1,3), (1,4), (1,5)])
for i in range(0, 5):
  y[i] = x[i][0] * 2 + x[i][1] * 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(4, input_shape=(2,)))
model.add(keras.layers.Dense(3))
model.add(keras.layers.Dense(1))
model.compile('SGD', 'mse')

print('x:', x[2:])

model.fit(x[:2], y[:2], epochs=10000, verbose=0)

print('Targets:', y[2:])
print('Predictions:', model.predict(x[2:,:]))



###########################################################################################################################################################
##############################################
# Modeling
##############################################
from keras import layers, models


def ANN_models_func(Nin, Nh, Nout):
    x = layers.Input(shape=(Nin,))
    h = layers.Activation('relu')(layers.Dense(Nh)(x))
    y = layers.Activation('softmax')(layers.Dense(Nout)(h))
    model = models.Model(x, y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def ANN_seq_func(Nin, Nh, Nout):
    model = models.Sequential()
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
    model.add(layers.Dense(Nout, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


class ANN_models_class(models.Model):
    def __init__(self, Nin, Nh, Nout):
        # Prepare network layers and activate functions
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        # Connect network elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


##############################################
# Data
##############################################
import numpy as np
from keras import datasets  # mnist
from keras.utils import np_utils  # to_categorical


def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)


##############################################
# Plotting
##############################################
import matplotlib.pyplot as plt


def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
    # plt.show()


def plot_loss(history, title=None):
    # summarize history for loss
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
    # plt.show()


##############################################
# Main
##############################################
def main():
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class

    # model = ANN_models_func(Nin, Nh, Nout)
    # model = ANN_models_class(Nin, Nh, Nout)
    model = ANN_seq_class(Nin, Nh, Nout)
    (X_train, Y_train), (X_test, Y_test) = Data_func()

    ##############################################
    # Training
    ##############################################
    history = model.fit(X_train, Y_train, epochs=15, batch_size=100, validation_split=0.2)
    performace_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy ->', performace_test)

    plot_loss(history)
    plt.show()
    #plot_acc(history)
    #plt.show()


# Run code
if __name__ == '__main__':
    main()



###########################################################################################################################################################
# 텐서보드 활용해보기
from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

# Load the TensorBoard notebook extension
%load_ext tensorboard

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

tf_callback = TensorBoard(log_dir="logs", histogram_freq=1)
model.fit(x_train, y_train, epochs=5, callbacks=[tf_callback])

model.evaluate(x_test,  y_test, verbose=2)


###########################################################################################################################################################
# 텐서보드 표출
%tensorboard --logdir logs


###########################################################################################################################################################
# 간단한 prediction 예제


import keras
import numpy

x = numpy.array([0, 1, 2, 3, 4])
y = x * 2 + 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x[:2], y[:2], epochs=1000, verbose=0)

print('Targets:', y[2:])
print('Predictions:', model.predict(x[2:]).flatten())



###########################################################################################################################################################
# 김태영 케라스
# https://tykimos.github.io/2017/02/04/MLP_Getting_Started/
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 랜덤시드 고정시키기
np.random.seed(5)

# 1. 데이터 준비하기
dataset = np.loadtxt("./warehouse/pima-indians-diabetes.data", delimiter=",")

# 2. 데이터셋 생성하기
x_train = dataset[:700,0:8]
y_train = dataset[:700,8]
x_test = dataset[700:,0:8]
y_test = dataset[700:,8]

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 4. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습시키기
model.fit(x_train, y_train, epochs=1500, batch_size=64)

# 6. 모델 평가하기
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


###########################################################################################################################################################
# 강화학습 open ai 막대세우기

# https://github.com/dhrim/keras_example_seminia_2020/blob/master/actor_critic_cartpole.ipynb

###########################################################################################################################################################
# rnn exam
# https://github.com/alucard001/RNN-Example/blob/master/Keras%20RNN%20Example.ipynb

from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

import numpy as np


values = np.random.random((100, 2))

input_train, target_train = values[0:70, 0], values[0:70, 1]
input_test , target_test  = values[70:, 0], values[70:, 1]

input_train = input_train.reshape(len(input_train), 1, 1)
input_test = input_test.reshape(len(input_test), 1, 1)

epochs = 10
batchsize = 10
timestep = 1
input_dim = 1

batchsize

# Build model
model = Sequential()
model.add(LSTM(10, activation='sigmoid', batch_input_shape=(batchsize, timestep, input_dim), return_sequences=True, stateful=True))
model.add(Dropout(0.5))
model.add(LSTM(8, stateful=True))
model.add(Dropout(0.5))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='rmsprop',
              loss='mse', metrics=['accuracy'])


# history = model.fit(inputs, y, epochs=100, batch_size=len(values), verbose=0)
history = model.fit(input_train, target_train, epochs=epochs, 
                   batch_size=batchsize, shuffle=False,
                   validation_data=(input_test, target_test))


# new_values = np.random.random(100).reshape(100, 1, 1)
predict = model.predict(input_test, batch_size=batchsize)

import matplotlib.pyplot as plt 
plt.plot(input_test.reshape(30,1), color='red', label="input test")
plt.plot(predict.reshape(30,1), color='blue', label="prediction")
plt.plot(target_test.reshape(30,1), color='orange', label="target test")
plt.title("Random RNN")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

###########################################################################################################################################################




import pandas as pd
import matplotlib.pyplot as plt
import datetime
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from Investar import Analyzer
import numpy as np

mk = Analyzer.MarketDB()

for keys in mk.codes:

    print(f"keys:{keys} company:{mk.codes[f'{keys}']}")
    
df = mk.get_daily_price('삼성전자', '2021-06-18')
print(f"df:{df}")
