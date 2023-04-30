# Bitstring

### ARCHITECTURE

- number of hidden layers [2 bits: 00 is one, 01 is two, 10 is three, 11 is four]
- kind of hidden layers [2 bits: 00 is dense, 01 is convolutional, 10 is pooling, 11 is invalid(?)]
- number of neurons per layer [3 bits: 000 is 2^1, 001 is 2^2, 010 is 2^3, 011 is 2^4, 100 is 2^5, 101 is 2^6, 110 is 2^7, 111 is 2^8]
- activation function for hidden layers [2 bits: 00 for linear, 01 for relu, 10 for sigmoid, 11 for tanh]
- activation function for output layer [2 bits: same as above]

### HYPERPARAMETERS

- learning rate [2 bits: 00 is 10^-2, 01 is 10^-3, 10 is 10^-4, 11 is 10^-5]
- decay rate [2 bits: 00 is "no decay", 01 is .99, 10 is .9, 11 is .85]
- optimizer [1 bit: 0 is SGD, 1 is Adam]

## EXAMPLE

example bitstring would be: [01_00_010_01_10_00_11_1]

This bitstring would equate to a model with:

- 2 dense hidden layers with 2^3 (8) neurons each
- relu activation function for hidden layers
- sigmoid activation function for output layer
- Adam optimizer with learning rate of 10^-2 and lr decay rate of .85
