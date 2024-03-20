from network import Network
from tqdm import tqdm


def train_networks(networks, dataset, size_of_train, size_of_test):
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset, size_of_train=size_of_train,size_of_test=size_of_test)
        network.print_network()
        pbar.update(1)
    pbar.close()

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])


def print_networks(networks):
    print('-' * 80)
    for network in networks:
        network.print_network()


def generate_network_list(nn_param_choices):
    networks = []

    # This is silly.
    for nbn in nn_param_choices['nb_neurons']:
        for nbl in nn_param_choices['nb_layers']:
            for a in nn_param_choices['activation']:
                for o in nn_param_choices['optimizer']:
                    # Set the parameters.
                    network = {
                        'nb_neurons': nbn,
                        'nb_layers': nbl,
                        'activation': a,
                        'optimizer': o,
                    }

                    # Instantiate a network object with set parameters.
                    network_obj = Network()
                    network_obj.create_set(network)

                    networks.append(network_obj)

    return networks


if __name__ == '__main__':
    dataset = input("Enter the dataset [Options: 'mnist', 'cifar10']\n")
    size_of_train = int(input("Enter the amount of pictures to train on [Defaults: MNIST-60000, CIFAR10-50000]\n"))
    size_of_test = int(input("Enter the amount of pictures to test on [Defaults: MNIST-10000, CIFAR10-10000]\n"))
    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'optimizer': ['lbfgs', 'adam', 'sgd'],
    }

    print("***Brute forcing networks***")

    networks = generate_network_list(nn_param_choices)

    train_networks(networks, dataset, size_of_train, size_of_test)
