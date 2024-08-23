import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.dists import Choice
from nengo.processes import Piecewise


def multiply(x):
    return x[0] * x[1]


model = nengo.Network(label="Multiplication")
with model:
    # Create 4 ensembles of leaky integrate-and-fire neurons
    A = nengo.Ensemble(100, dimensions=1, radius=10)
    B = nengo.Ensemble(100, dimensions=1, radius=10)
    combined = nengo.Ensemble(
        220, dimensions=2, radius=15
    )  # This radius is ~sqrt(10^2+10^2)
    prod = nengo.Ensemble(100, dimensions=1, radius=20)

    combined.encoders = Choice([[1, 1], [-1, 1], [1, -1], [-1, -1]])

    inputA = nengo.Node(Piecewise({0: 0, 2.5: 10, 4: -10}))
    inputB = nengo.Node(Piecewise({0: 10, 1.5: 2, 3: 0, 4.5: 2}))

    correct_ans = Piecewise({0: 0, 1.5: 0, 2.5: 20, 3: 0, 4: 0, 4.5: -20})

    nengo.Connection(inputA, A)
    nengo.Connection(inputB, B)

    nengo.Connection(A, combined[1])
    nengo.Connection(B, combined[0])

    nengo.Connection(combined, prod, function=multiply)

    inputA_probe = nengo.Probe(inputA)
    inputB_probe = nengo.Probe(inputB)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)
    combined_probe = nengo.Probe(combined, synapse=0.01)
    prod_probe = nengo.Probe(prod)

    # Create the simulator
    with nengo.Simulator(model) as sim:
        # Run it for 5 seconds
        sim.run(5)

        plt.figure()
        plt.plot(sim.trange(), sim.data[A_probe], label="Decoded A")
        plt.plot(sim.trange(), sim.data[B_probe], label="Decoded B")
        plt.plot(sim.trange(), sim.data[prod_probe], label="Decoded product")
        plt.plot(
            sim.trange(), correct_ans.run(sim.time, dt=sim.dt), c="k", label="Actual product"
        )
        plt.legend(loc="best")
        plt.ylim(-25, 25)

        plt.show()
