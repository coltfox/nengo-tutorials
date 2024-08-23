import matplotlib.pyplot as plt
import nengo

model = nengo.Network(label="Addition")
with model:
    A = nengo.Ensemble(100, dimensions=1)
    B = nengo.Ensemble(100, dimensions=1)
    C = nengo.Ensemble(100, dimensions=1)

    input_a = nengo.Node(output=0.5)
    input_b = nengo.Node(output=0.3)

    nengo.Connection(input_a, A)
    nengo.Connection(input_b, B)

    nengo.Connection(A, C)
    nengo.Connection(B, C)

    input_a_probe = nengo.Probe(input_a)
    input_b_probe = nengo.Probe(input_b)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)
    C_probe = nengo.Probe(C, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(5)

        t = sim.trange()
        plt.figure()
        plt.plot(sim.trange(), sim.data[A_probe], label="Decoded Ensemble A")
        plt.plot(sim.trange(), sim.data[B_probe], label="Decoded Ensemble B")
        plt.plot(sim.trange(), sim.data[C_probe], label="Decoded Ensemble C")
        plt.plot(
            sim.trange(), sim.data[input_a_probe], label="Input A", color="k", linewidth=2.0
        )
        plt.plot(
            sim.trange(), sim.data[input_b_probe], label="Input B", color="0.75", linewidth=2.0
        )
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel("time [s]")

        plt.show()
