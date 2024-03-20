# genetic-algorithm-for-neural-network-params
Project in AI consisting of a Genetic Algorithm for optimizing parameters of Neural Network. The program takes long to run with our limited resources and because of that only the MNIST dataset was used for training and testing.

**Done in collaboration with**:
- Tamara Stojanova https://github.com/tamarastojanova
- Andrej Sterjev
- Sonja Petrovska

**Introduction**

Genetic algorithms are one of the most commonly used families of optimization algorithms in the field of artificial intelligence. Artificial neural networks are among the most powerful models that can be used for any type of machine learning. However, one of the main challenges in using neural networks to solve machine learning problems is the uncertainty of what the optimal architecture for the neural network is for returning the best possible results, with the highest precision and without overfitting. This is something that cannot be known in advance.

In this term paper, we explore the potential of combining neural networks with genetic algorithms, by using an appropriate representation of neural networks as samples from a population whose characteristics the genetic algorithm will be able to optimize.

**Results and conclusion**

From this project, we can conclude that we got an excellent result of 98.42% accuracy by applying the genetic algorithm on the neural network. This result matches the theoretically best result of the program with the possible parameters, which the brute-force algorithm found. However, the best result may not be obtained every time we run the program with the genetic algorithm. Sometimes we may get worse results, but in most cases they will be only slightly worse than the best results. That is because we do not examine all combinations of parameters in the genetic algorithm, i.e. it may happen that we do not generate the exact combination that would give us the best results. Also, the combinations are generated and intersected randomly, which is clearly not the same in every run of the program.

The genetic algorithm runs 14 hours and 43 minutes and generates and trains 200 networks, while the brute force algorithm runs for almost 5 days and generates and trains 240 networks, so one might wonder how that is possible. This is because in the genetic algorithm, we allow the same neural networks to be created but not trained. There is a chance to get networks with the same parameters throughout the different generations due to the very nature of selecting parameters in a random way, and even more so due to the small selection of possible parameters that we have at our disposal. This number is small mainly due to limited resources on our side in terms of performance, but also due to the sklearn MLPClassifier model itself, which does not have much choice for possible parameters. As mentioned, those networks that have been already trained we do not retrain, because it is redundant and in turn significantly shortens the execution time.

According to some data found online, human performance on the MNIST dataset leads to an average accuracy of 99.77%. This means that our algorithm has not yet "beat" the human. However, in our code we have used models that we have studied and are well-known to us, and they unfortunately do not have many possible options for
the parameters. If we used some models with a wider range of parameters and their values, we might even get an accuracy higher than that of humans.
