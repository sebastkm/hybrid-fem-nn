# Dense neural networks in UFL

In this repository you can find an implementation for fully connected neural networks in UFL.
This implementation was used for some examples in the paper "[Hybrid FEM-NN models: Combining artificial neural networks with the finite element method](https://arxiv.org/abs/2101.00962)".

## Usage

The neural network is defined through the `ANN` class, which is imported by
```
from ufl_dnn.neural_network import ANN
```
FEniCS and dolfin-adjoint must also be imported in order to enable automatic differentiation:
```
from fenics import *
from dolfin_adjoint import *
```

The ANN class has two required keyword arguments `layers` and `bias`.
`layers` is a list of the number of outputs of each layer, with the first element being the number of inputs to the neural network.
`bias` is a list of booleans of length `len(layers)-1` specifying if each layer has an additive bias.
```
layers = [3, 10, 1]
bias = [True, True]
net = ANN(layers, bias=bias)
```
By default the activation function is `ufl.tanh`, but a different one can be specified through the `sigma` keyword argument.

Once the network is defined, it acts as a function that returns a UFL expression representing the neural network.
For a spatially varying neural network, it can be used as follows
```
mesh = UnitSquareMesh(Nx, Ny)
x, y = SpatialCoordinate(mesh)
form = net(x, y) * dx
```
or with a function
```
f = Function(V)
v = TestFunction(V)

form = net(x, y, f) * v * dx
```

Dolfin-adjoint can be used to optimize the weight of the network.
Once you have trained your network, you can save it to a file through `pickle`.
Saving and loading can be done as follows:
```
# Save network
net.save("trained_network.pkl)

# Load network
net = ANN("trained_network.pkl")
```

For more complete code examples, see the `tests` folder.

## Installation
This package requires [FEniCS](https://fenicsproject.org/) and [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) in order to work.
The package itself can be installed with pip after cloning this repo
```
pip install .
```
or directly
```
pip install git+https://github.com/sebastkm/hybrid-fem-nn.git@master
```