"""
Public API for GPU-accelerated CTRNN and Izhikevich spiking evaluation.

Usage::

    from neat.gpu.evaluator import GPUCTRNNEvaluator

    def input_fn(t, dt):
        return [math.sin(t), math.cos(t)]

    def fitness_fn(output_trajectory):
        # output_trajectory: ndarray [num_steps, num_outputs]
        return float(output_trajectory[-1, 0])

    evaluator = GPUCTRNNEvaluator(dt=0.01, t_max=1.0,
                                   input_fn=input_fn,
                                   fitness_fn=fitness_fn)
    pop.run(evaluator.evaluate, n=100)
"""

from neat.gpu import _import_numpy


class GPUCTRNNEvaluator:
    """
    GPU-accelerated parallel evaluator for CTRNN networks.

    Drop-in replacement for ``neat.ParallelEvaluator`` — pass
    ``evaluator.evaluate`` to ``Population.run()``.

    Parameters
    ----------
    dt : float
        Integration time step (seconds).
    t_max : float
        Total simulation time (seconds).
    input_fn : callable
        ``input_fn(t, dt) -> array-like [num_inputs]`` or
        ``input_fn(t, dt) -> array-like [N, num_inputs]`` for per-genome inputs.
        Called once per step on CPU to produce the input signal.
    fitness_fn : callable
        ``fitness_fn(output_trajectory) -> float`` where output_trajectory
        is an ndarray of shape ``[num_steps, num_outputs]``.
        Called once per genome on CPU after GPU simulation.
    """

    def __init__(self, dt, t_max, input_fn, fitness_fn):
        self.dt = dt
        self.t_max = t_max
        self.input_fn = input_fn
        self.fitness_fn = fitness_fn

    def evaluate(self, genomes, config):
        """
        Evaluate all genomes in the population on GPU.

        This method has the same signature as a NEAT fitness function:
        ``evaluate(genomes, config)`` where genomes is a list of
        ``(genome_id, genome)`` tuples. It assigns ``genome.fitness``
        for each genome.
        """
        np = _import_numpy()
        # Lazy import to avoid loading CuPy at module import time.
        from neat.gpu._padding import pack_ctrnn_population
        from neat.gpu._cupy_backend import evaluate_ctrnn_batch

        num_steps = int(self.t_max / self.dt)

        # Precompute input trajectory on CPU.
        # first_input shape is either [num_inputs] or [N, num_inputs].
        first_input = np.asarray(self.input_fn(0.0, self.dt), dtype=np.float32)
        inputs = np.zeros((num_steps, *first_input.shape), dtype=np.float32)
        inputs[0] = first_input
        for step in range(1, num_steps):
            inputs[step] = np.asarray(
                self.input_fn(step * self.dt, self.dt), dtype=np.float32)

        # Pack genomes into padded arrays.
        packed = pack_ctrnn_population(genomes, config)

        # Run GPU simulation.
        trajectory = evaluate_ctrnn_batch(packed, inputs, self.dt)
        # trajectory: [N, num_steps, num_outputs]

        # Assign fitness from CPU-side fitness function.
        for i, (genome_id, genome) in enumerate(genomes):
            genome.fitness = self.fitness_fn(trajectory[i])


class GPUIZNNEvaluator:
    """
    GPU-accelerated parallel evaluator for Izhikevich spiking networks.

    Parameters
    ----------
    dt : float
        Integration time step (milliseconds).
    t_max : float
        Total simulation time (milliseconds).
    input_fn : callable
        ``input_fn(t, dt) -> array-like [num_inputs]`` or per-genome variant.
        Returns input values (not spikes) for input pins at time t.
    fitness_fn : callable
        ``fitness_fn(output_trajectory) -> float`` where output_trajectory
        is an ndarray of shape ``[num_steps, num_outputs]`` containing
        spike indicators (0.0 or 1.0).
    """

    def __init__(self, dt, t_max, input_fn, fitness_fn):
        self.dt = dt
        self.t_max = t_max
        self.input_fn = input_fn
        self.fitness_fn = fitness_fn

    def evaluate(self, genomes, config):
        """Evaluate all genomes on GPU. Same interface as NEAT fitness function."""
        np = _import_numpy()
        from neat.gpu._padding import pack_iznn_population
        from neat.gpu._cupy_backend import evaluate_iznn_batch

        num_steps = int(self.t_max / self.dt)

        # Precompute input trajectory on CPU.
        # first_input shape is either [num_inputs] or [N, num_inputs].
        first_input = np.asarray(self.input_fn(0.0, self.dt), dtype=np.float32)
        inputs = np.zeros((num_steps, *first_input.shape), dtype=np.float32)
        inputs[0] = first_input
        for step in range(1, num_steps):
            inputs[step] = np.asarray(
                self.input_fn(step * self.dt, self.dt), dtype=np.float32)

        packed = pack_iznn_population(genomes, config)
        trajectory = evaluate_iznn_batch(packed, inputs, self.dt, num_steps)

        for i, (genome_id, genome) in enumerate(genomes):
            genome.fitness = self.fitness_fn(trajectory[i])
