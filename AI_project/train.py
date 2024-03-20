import numpy as np
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier


def transform_to_float_and_divide(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, x_test


def get_desired_amount_of_data(x_train, y_train, x_test, y_test, size_of_train, size_of_test):
    x_train = x_train[:size_of_train]
    y_train = y_train[:size_of_train]
    x_test = x_test[:size_of_test]
    y_test = y_test[:size_of_test]
    return x_train, y_train, x_test, y_test


def get_cifar10(size_of_train, size_of_test):
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train, x_test, y_test = get_desired_amount_of_data(x_train, y_train, x_test, y_test, size_of_train,
                                                                  size_of_test)
    x_train = x_train.reshape(size_of_train, 3072)
    x_test = x_test.reshape(size_of_test, 3072)
    x_train, x_test = transform_to_float_and_divide(x_train, x_test)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    return batch_size, x_train, x_test, y_train, y_test


def get_mnist(size_of_train, size_of_test):
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train, x_test, y_test = get_desired_amount_of_data(x_train, y_train, x_test, y_test, size_of_train,
                                                                  size_of_test)
    x_train = x_train.reshape(size_of_train, 784)
    x_test = x_test.reshape(size_of_test, 784)
    x_train, x_test = transform_to_float_and_divide(x_train, x_test)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    return batch_size, x_train, x_test, y_train, y_test


def compile_model(network, batch_size):
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    list_layers = []
    for i in range(nb_layers):
        list_layers.append(nb_neurons)
    layers_with_neurons = tuple(list_layers)
    model = MLPClassifier(hidden_layer_sizes=layers_with_neurons, activation=activation, solver=optimizer,
                          max_iter=10000, verbose=0, batch_size=batch_size, random_state=0)
    return model


def train_and_find_accuracy(network, dataset, size_of_train, size_of_test):
    if dataset == 'cifar10':
        batch_size, x_train, x_test, y_train, y_test = get_cifar10(size_of_train, size_of_test)
    elif dataset == 'mnist':
        batch_size, x_train, x_test, y_train, y_test = get_mnist(size_of_train, size_of_test)
    model = compile_model(network, batch_size)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    correctly_predicted = 0
    for true, pred in zip(y_test, predictions):
        pred = np.array(pred, dtype=float)
        element_wise_comparison = pred == true
        if np.all(element_wise_comparison):
            correctly_predicted += 1
    score = correctly_predicted / len(y_test)
    return score
