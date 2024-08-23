import matplotlib.pyplot as plt
import numpy as np

import nengo

model = nengo.Network(label='Communications Channel')
with model:
    sin = nengo.Node(np.sin)

    A = nengo.Ensemble(100, dimensions=1)
    B = nengo.Ensemble(100, dimensions=1)

    nengo.Connection(sin, A)

    nengo.Connection(A, B)

    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(2)

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.title("Input")
        plt.plot(sim.trange(), sim.data[sin_probe])
        plt.ylim(0, 1.2)
        plt.subplot(1, 3, 2)
        plt.title("A")
        plt.plot(sim.trange(), sim.data[A_probe])
        plt.ylim(0, 1.2)
        plt.subplot(1, 3, 3)
        plt.title("B")
        plt.plot(sim.trange(), sim.data[B_probe])
        plt.ylim(0, 1.2)

        plt.show()
