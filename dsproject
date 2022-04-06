from flask import Flask
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils

app = Flask(__name__)

model = load_model('./mini_test_model.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

@app.route("/")
def hello_world():
    predictions = model.predict(x_test)
    correct_predictions = np.argmax(predictions, axis = 1)
    labels = np.argmax(y_test, axis = 1)
    correct_6 = 0
    wrong_6 = 0
    
    for i in range(len(correct_predictions)):
        if correct_predictions[i] == labels[i]:
            correct_6 += 1
        else:
            wrong_6 += 1
    
    print(correct_6,'classified correctly')
    print(wrong_6,'classified incorrectly')
    return "<p>Hello, World!</p>"

if __name__ == '__main__':
    app.run()
