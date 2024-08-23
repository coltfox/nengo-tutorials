import matplotlib.pyplot as plt
import numpy as np

import nengo

model = nengo.Network(label="2D Representation")
with model:
    neurons = nengo.Ensemble(100, dimensions=2)

    sin = nengo.Node(output=np.sin)
    cos = nengo.Node(output=np.cos)

    nengo.Connection(sin, neurons[0])
    nengo.Connection(cos, neurons[1])

    sin_probe = nengo.Probe(sin, "output")
    cos_probe = nengo.Probe(cos, "output")
    neurons_probe = nengo.Probe(neurons, "decoded_output", synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(5)

    plt.figure()
    plt.plot(sim.trange(), sim.data[neurons_probe], label="Decoded output")
    plt.plot(sim.trange(), sim.data[sin_probe], "r", label="Sine")
    plt.plot(sim.trange(), sim.data[cos_probe], "k", label="Cosine")
    plt.legend()
    plt.xlabel("time [s]")

    plt.show()
