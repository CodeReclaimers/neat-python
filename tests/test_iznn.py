import neat
import pytest


def _make_neuron(params, bias=0.0, inputs=None):
    if inputs is None:
        inputs = []
    return neat.iznn.IZNeuron(bias, params["a"], params["b"], params["c"], params["d"], inputs)


def test_single_neuron_regular_spiking_pulse_response():
    """Basic dynamic test mirroring the demo-iznn pulse protocol."""
    params = neat.iznn.REGULAR_SPIKING_PARAMS
    n = _make_neuron(params, bias=0.0)

    dt = 0.25
    spikes = []

    # Drive the neuron with a pulse of current between 100 and 800 steps,
    # following the example in examples/neuron-demo/demo-iznn.py.
    for step in range(1000):
        n.current = 0.0 if step < 100 or step > 800 else 10.0
        spikes.append(n.fired)
        n.advance(dt)

    # No spikes without input current.
    assert max(spikes[:100]) == 0.0

    # The neuron should spike at least once while being driven.
    assert max(spikes[100:800]) == 1.0

    # After the drive is removed the neuron should eventually become quiet again.
    # Allow some transient after the input turns off.
    assert max(spikes[900:]) == 0.0

    # Membrane potential should relax back near the reset value c.
    # Use a loose bound here since the exact value depends on integration details.
    assert abs(n.v - params["c"]) < 20.0


def test_izneuron_reset_restores_initial_state():
    """IZNeuron.reset should restore v, u, fired, and current to defaults."""
    params = neat.iznn.FAST_SPIKING_PARAMS
    n = _make_neuron(params, bias=1.5)

    # Perturb state.
    n.current = 3.0
    n.advance(0.5)

    # Also modify outputs explicitly so reset has work to do.
    n.fired = 1.0
    n.current = 7.0

    n.reset()

    assert n.v == n.c
    assert n.u == n.b * n.v
    assert n.fired == 0.0
    assert n.current == n.bias


def test_iznn_uses_external_inputs_and_resets_neurons():
    """IZNN should aggregate external inputs correctly and support reset()."""
    params = neat.iznn.REGULAR_SPIKING_PARAMS

    # Single output neuron (key 0) receiving two external inputs -1 and -2.
    neuron = _make_neuron(params, bias=0.0, inputs=[(-1, 1.0), (-2, -1.0)])
    neurons = {0: neuron}
    inputs = [-1, -2]
    outputs = [0]

    net = neat.iznn.IZNN(neurons, inputs, outputs)

    # With zero inputs the synaptic current should equal the bias.
    net.set_inputs([0.0, 0.0])
    net.advance(0.25)
    assert neuron.current == pytest.approx(neuron.bias)

    # A positive value on input -1 and zero on -2 should increase current.
    net.reset()
    net.set_inputs([1.0, 0.0])
    net.advance(0.25)
    assert neuron.current == pytest.approx(neuron.bias + 1.0)

    # A positive value on input -2 (with negative weight) should decrease current.
    net.reset()
    net.set_inputs([0.0, 1.0])
    net.advance(0.25)
    assert neuron.current == pytest.approx(neuron.bias - 1.0)

    # Reset should restore neuron state for all neurons in the network.
    neuron.v = 0.0
    neuron.u = 0.0
    neuron.current = 10.0
    neuron.fired = 1.0

    net.reset()

    assert neuron.v == neuron.c
    assert neuron.u == neuron.b * neuron.v
    assert neuron.current == neuron.bias
    assert neuron.fired == 0.0


def test_iznn_set_inputs_length_mismatch_raises():
    """set_inputs should enforce input length and raise RuntimeError on mismatch."""
    params = neat.iznn.REGULAR_SPIKING_PARAMS
    neuron = _make_neuron(params)
    neurons = {0: neuron}
    inputs = [-1, -2]
    outputs = [0]
    net = neat.iznn.IZNN(neurons, inputs, outputs)

    # Too few inputs.
    with pytest.raises(RuntimeError, match="Number of inputs"):
        net.set_inputs([1.0])

    # Too many inputs.
    with pytest.raises(RuntimeError, match="Number of inputs"):
        net.set_inputs([1.0, 2.0, 3.0])


def test_get_time_step_positive():
    """get_time_step_msec should return a positive float."""
    params = neat.iznn.REGULAR_SPIKING_PARAMS
    neuron = _make_neuron(params)
    net = neat.iznn.IZNN({0: neuron}, inputs=[-1], outputs=[0])

    dt = net.get_time_step_msec()
    assert isinstance(dt, float)
    assert dt > 0.0
