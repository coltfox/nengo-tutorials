import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot

model = nengo.Network(label="Two neurons")

with model:
    neurons = nengo.Ensemble(
        2,
        dimensions=1,
        intercepts=Uniform(-0.5, -0.5),
        max_rates=Uniform(100, 100),
        encoders=[[1], [-1]]
    )

    sin = nengo.Node(lambda t: np.sin(8 * t))

    nengo.Connection(sin, neurons, synapse=0.01)

    sin_probe = nengo.Probe(sin)
    spikes = nengo.Probe(neurons.neurons)
    voltage = nengo.Probe(neurons.neurons, "voltage")
    filtered = nengo.Probe(neurons, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(1)

    t = sim.trange()

    # Plot the decoded output of the ensemble
    plt.figure()
    plt.plot(t, sim.data[filtered])
    plt.plot(t, sim.data[sin_probe])
    plt.xlim(0, 1)

    # Plot the spiking output of the ensemble
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    rasterplot(t, sim.data[spikes], colors=[(1, 0, 0), (0, 0, 0)])
    plt.yticks((1, 2), ("On neuron", "Off neuron"))
    plt.ylim(2.5, 0.5)

    # Plot the soma voltages of the neurons
    plt.subplot(2, 2, 2)
    plt.plot(t, sim.data[voltage][:, 0] + 1, "r")
    plt.plot(t, sim.data[voltage][:, 1], "k")
    plt.yticks(())
    plt.axis([0, 1, 0, 2])
    plt.subplots_adjust(wspace=0.05)

    plt.show()
