import matplotlib.pyplot as plt
import numpy as np

import nengo

model = nengo.Network(label='Squaring')


def square(x):
    return x[0] * x[0]


with model:
    A = nengo.Ensemble(100, dimensions=1)
    B = nengo.Ensemble(100, dimensions=1)

    sin = nengo.Node(np.sin)

    nengo.Connection(sin, A)
    nengo.Connection(A, B, function=square)

    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(5)

        # Plot the input signal and decoded ensemble values
        plt.figure()
        plt.plot(sim.trange(), sim.data[A_probe], label="Decoded Ensemble A")
        plt.plot(sim.trange(), sim.data[B_probe], label="Decoded Ensemble B")
        plt.plot(
            sim.trange(), sim.data[sin_probe], label="Input Sine Wave", color="k", linewidth=2.0
        )
        plt.legend(loc="best")
        plt.ylim(-1.2, 1.2)

        plt.show()
