import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot

model = nengo.Network(label="Many neurons")

with model:
    A = nengo.Ensemble(100, dimensions=1)

    sin = nengo.Node(lambda t: np.sin(8 * t))

    nengo.Connection(sin, A, synapse=0.01)

    sin_probe = nengo.Probe(sin)
    A_spikes = nengo.Probe(A.neurons)
    A_probe = nengo.Probe(A, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(1)

    # Plot the decoded output of the ensemble
    plt.figure()
    plt.plot(sim.trange(), sim.data[A_probe], label="A output")
    plt.plot(sim.trange(), sim.data[sin_probe], "r", label="Input")
    plt.xlim(0, 1)
    plt.legend()

    # Plot the spiking output of the ensemble
    plt.figure()
    rasterplot(sim.trange(), sim.data[A_spikes])
    plt.xlim(0, 1)

    plt.show()
