import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import network
from keras.datasets import mnist

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(3, 3)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    # plt.savefig("result_mnist.png")
    plt.show()


def preprocessing(d):
    w, h = d.shape
    threshold = threshold_mean(d)
    # print(threshold)
    binary_value = d > threshold #returns True or False
    shift = 2*(binary_value*1)-1 # Boolean to int conversion
    # print(shift)
    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()
data = []
for i in range(2):
    xi = x_train[y_train==i]
    # print(xi)
    data.append(xi[0])
# print(data)

#data preprocessing
print("Start data preprocessing...")
data = [preprocessing(d) for d in data]

# Create Hopfield Network Model
model = network.HopfieldNetwork()
model.train_weights(data)

#create test data
test = []
for i in range(2):
    xi = x_train[y_train==i]
    test.append(xi[1])
test = [preprocessing(d) for d in test]

predicted = model.predict(test, threshold=50)
print("Show prediction results...")
plot(data, test, predicted, figsize=(5, 5))
