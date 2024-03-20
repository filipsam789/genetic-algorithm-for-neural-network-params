import random
from train import train_and_find_accuracy


class Network:
    def __init__(self, nn_param_choices=None):
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        self.network = network

    def train(self, dataset, size_of_train, size_of_test):
        if self.accuracy == 0.:
            self.accuracy = train_and_find_accuracy(self.network, dataset, size_of_train, size_of_test)

    def print_network(self):
        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
