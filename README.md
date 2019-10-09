# Feed_Forward_network_C-
You’ll read from a file: the number of layers, number of neurons per each layer, the
values of the input layer and the weights.
File format:
1. First line of input contains the number of layers followed by numbers of
neurons per each layer.
* Second line contains the input layer.
* Next there will be a line per every neuron in the network, except for the output
layer.
* Each of the following lines will contain the weights on the connections
between the neuron and all the neurons in the next layer.
Here’s an example of a file that contains the neural network in figure 1:
* 3 3 2 3
* 1 2 3
* 0.12 0.54
* 0.87 0.34
* 0.69 0.2
* 0.31 0.123 0.154
* 0.77 0.34 0.456
* expected output :
* 2.6197 1.10219 1.43514
