from optimizer import Optimizer
from tqdm import tqdm


def train_networks(networks, networks_dict, dataset, size_of_train, size_of_test):
    pbar = tqdm(total=len(networks))
    for network in networks:
        if networks_dict[str(network.network)] == 0:
            network.train(dataset, size_of_train, size_of_test)
            networks_dict[str(network.network)] = network.accuracy
        else:
            network.accuracy = networks_dict[str(network.network)]
        pbar.update(1)
    pbar.close()


def get_average_accuracy(networks):
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy
    return total_accuracy / len(networks)


def generate(generations, population, nn_param_choices, dataset, size_of_train, size_of_test):
    optimizer = Optimizer(nn_param_choices)
    networks, networks_dict = optimizer.create_population(population)
    # Evolve the generation.
    for i in range(generations):
        print("***Doing generation %d of %d***" %
              (i + 1, generations))
        # Train and get accuracy for networks.
        train_networks(networks, networks_dict, dataset, size_of_train, size_of_test)
        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)
        # Print out the average accuracy each generation.
        print("Generation average: %.2f%%" % (average_accuracy * 100))
        print('-' * 80)
        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks, networks_dict = optimizer.evolve(networks, networks_dict)
    # Sort our final population.
    sorted_dicts = sorted(networks_dict.items(), key=lambda item: item[1], reverse=True)
    # Print out the top 5 networks.
    print_networks(sorted_dicts[:5])


def print_networks(network_dicts):
    print('-' * 80)
    for network in network_dicts:
        print(str(network[0]))
        print("Network accuracy: %.2f%%" % (network[1] * 100))


if __name__ == '__main__':
    generations = int(input("Enter the number of generations, i.e. the number of times to evolve the population\n"))
    population = int(input("Enter the size of the population, i.e. the number of networks in each generation\n"))
    dataset = input("Enter the dataset [Options: 'mnist', 'cifar10']\n")
    size_of_train = int(input("Enter the amount of pictures to train on [Defaults: MNIST-60000, CIFAR10-50000]\n"))
    size_of_test = int(input("Enter the amount of pictures to test on [Defaults: MNIST-10000, CIFAR10-10000]\n"))
    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'optimizer': ['lbfgs', 'adam', 'sgd'],
    }

    print("***Evolving %d generations with population %d***" %
          (generations, population))

    generate(generations, population, nn_param_choices, dataset, size_of_train, size_of_test)
