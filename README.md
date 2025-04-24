# simple-neural-network

This is a simple feedforward neural network written in C++. Features include:
- Loading MNIST data (data files are included in project)
- Fully matrix-based feedforwarding and backpropagation
- Neural dropout
- L2 regularization
- Simple linear algebra functions (could perhaps be replaced with a dedicated library)

# Sample output

```
Nbr of training images = 60000
Nbr of training labels = 60000
Nbr of test images = 10000
Nbr of test labels = 10000
Epoch 0: 9414 / 10000
Epoch 1: 9505 / 10000
Epoch 2: 9559 / 10000
...
Epoch 14: 9615 / 10000
Epoch 15: 9645 / 10000
...
```

As we can see, with 15 epochs the net achieved a 96.45% accuracy in classifying digits in the MNIST database.

# Building

The project is quite self-contained; it requires a [MNIST database reader library](https://github.com/wichtounet/mnist) which is included among the project files. MNIST data files are also included, though they need to be unzipped to the project directory. The project can be compiled using the makefile provided; a binary compiled on Linux Mint 21.3 x86-64 is included among the project files for convenience.
