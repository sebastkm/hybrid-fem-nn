from fenics import *
from fenics_adjoint import *

from ufl_dnn.neural_network import ANN

from numpy.random import seed, randn
seed(1)


def test_optim():
    layers = [2, 3, 1]
    bias = [True, True]
    net = ANN(layers, bias=bias)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    F = inner(grad(u), grad(v))*dx - net(x, y)*v*dx

    bc = DirichletBC(V, 1, "on_boundary")
    solve(F == 0, u, bc)

    J = assemble((u-Constant(1))**2*dx)
    Jhat = ReducedFunctional(J, net.weights_ctrls())

    opt = minimize(Jhat, options={"disp": True, "gtol": 1e-12, "ftol": 1e-12, "maxiter": 50})

    net.set_weights(opt)
    assert assemble(net(x, y)**2*dx) < 1e-6


def test_taylor():
    layers = [3, 10, 1]
    bias = [True, True]
    net = ANN(layers, bias=bias)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    F = inner(grad(u), grad(v))*dx - net(x, y, u)*v*dx

    bc = DirichletBC(V, 1, "on_boundary")
    solve(F == 0, u, bc)

    J = assemble((u-Constant(1))**2*dx)
    Jhat = ReducedFunctional(J, net.weights_ctrls())

    m = net.weights_flat()
    h = []
    for w in net.weights_flat():
        h.append(Constant(randn(*w.ufl_shape)))

    assert taylor_test(Jhat, m, h) > 1.9


def test_save_and_load(tmpdir):
    layers = [2, 10, 1]
    bias = [True, True]
    net = ANN(layers, bias=bias)

    net.save(tmpdir.strpath + "/test_case.pkl")
    new_net = ANN(tmpdir.strpath + "/test_case.pkl")

    mesh = UnitSquareMesh(10, 10)
    x, y = SpatialCoordinate(mesh)

    assert net != new_net
    assert assemble(net(x, y)**2*dx(domain=mesh)) == assemble(new_net(x, y)**2*dx(domain=mesh))

