import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot

from nengo.dists import Uniform

model = nengo.Network(label="A Single Neuron")

with model:
    neuron = nengo.Ensemble(
        1,
        dimensions=1,
        intercepts=Uniform(-0.5, -0.5),
        max_rates=Uniform(100, 100),
        encoders=[[1]]
    )

    cos = nengo.Node(lambda t: np.cos(8 * t))

    nengo.Connection(cos, neuron)

    cos_probe = nengo.Probe(cos)
    spikes = nengo.Probe(neuron.neurons)
    voltage = nengo.Probe(neuron.neurons, "voltage")
    filtered = nengo.Probe(neuron, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(1)

        # Plot the decoded output of the ensemble
        plt.figure()
        plt.plot(sim.trange(), sim.data[filtered])
        plt.plot(sim.trange(), sim.data[cos_probe])
        plt.xlim(0, 1)

        # Plot the spiking output of the ensemble
        plt.figure(figsize=(10, 8))
        plt.subplot(221)
        rasterplot(sim.trange(), sim.data[spikes])
        plt.ylabel("Neuron")
        plt.xlim(0, 1)

        # Plot the soma voltages of the neurons
        plt.subplot(222)
        plt.plot(sim.trange(), sim.data[voltage][:, 0], "r")
        plt.xlim(0, 1)

        plt.show()
