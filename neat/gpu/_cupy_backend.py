"""
CuPy-based GPU kernels for batched CTRNN and Izhikevich spiking evaluation.

This module is only imported when a GPU evaluator is instantiated.
"""

from neat.gpu import _import_cupy, _import_numpy


# ---------------------------------------------------------------------------
# Activation function elementwise kernel (CTRNN)
# ---------------------------------------------------------------------------
# Dispatches on integer activation_id per element. The formulas and clipping
# match neat-python's activations.py exactly.

_ACTIVATION_KERNEL_CODE = '''
extern "C" __global__
void apply_activation(
    const float* __restrict__ s,          // aggregated input [N*M]
    const float* __restrict__ bias,       // [N*M]
    const float* __restrict__ response,   // [N*M]
    const int*   __restrict__ act_id,     // [N*M]
    float*       __restrict__ z,          // output [N*M]
    int total
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    float x = bias[idx] + response[idx] * s[idx];
    int aid = act_id[idx];
    float result;

    if (aid == 0) {
        // sigmoid: 1/(1+exp(-clip(5*x, -60, 60)))
        float t = 5.0f * x;
        t = fminf(60.0f, fmaxf(-60.0f, t));
        result = 1.0f / (1.0f + expf(-t));
    } else if (aid == 1) {
        // tanh: tanh(clip(2.5*x, -60, 60))
        float t = 2.5f * x;
        t = fminf(60.0f, fmaxf(-60.0f, t));
        result = tanhf(t);
    } else if (aid == 2) {
        // relu
        result = fmaxf(0.0f, x);
    } else if (aid == 3) {
        // identity
        result = x;
    } else if (aid == 4) {
        // clamped
        result = fminf(1.0f, fmaxf(-1.0f, x));
    } else if (aid == 5) {
        // elu
        result = (x > 0.0f) ? x : expf(x) - 1.0f;
    } else if (aid == 6) {
        // softplus: 0.2*log(1+exp(clip(5*x, -60, 60)))
        float t = 5.0f * x;
        t = fminf(60.0f, fmaxf(-60.0f, t));
        result = 0.2f * logf(1.0f + expf(t));
    } else if (aid == 7) {
        // sin: sin(clip(5*x, -60, 60))
        float t = 5.0f * x;
        t = fminf(60.0f, fmaxf(-60.0f, t));
        result = sinf(t);
    } else if (aid == 8) {
        // gauss: exp(-5*clip(x,-3.4,3.4)^2)
        float t = fminf(3.4f, fmaxf(-3.4f, x));
        result = expf(-5.0f * t * t);
    } else if (aid == 9) {
        // abs
        result = fabsf(x);
    } else if (aid == 10) {
        // square
        result = x * x;
    } else {
        // fallback: identity
        result = x;
    }

    z[idx] = result;
}
'''


def _get_activation_kernel():
    """Compile and cache the activation kernel."""
    cp = _import_cupy()
    return cp.RawKernel(_ACTIVATION_KERNEL_CODE, 'apply_activation')


def evaluate_ctrnn_batch(packed, inputs_cpu, dt):
    """
    Run batched CTRNN simulation on GPU using exponential Euler integration.

    Parameters
    ----------
    packed : dict
        Output of pack_ctrnn_population(). NumPy arrays on CPU.
    inputs_cpu : ndarray [num_steps, num_inputs] or [num_steps, N, num_inputs]
        Precomputed input trajectory. If 2-D, broadcast across population.
    dt : float
        Integration time step.

    Returns
    -------
    trajectory : ndarray [N, num_steps, num_outputs] float32 on CPU
        Output node states at each time step.
    """
    cp = _import_cupy()
    np = _import_numpy()

    N = packed['W'].shape[0]
    M = packed['max_nodes']
    num_inputs = packed['num_inputs']
    num_outputs = packed['num_outputs']
    num_steps = inputs_cpu.shape[0]

    # Transfer parameters to GPU.
    W = cp.asarray(packed['W'])                     # [N, M, M]
    bias = cp.asarray(packed['bias'])               # [N, M]
    response = cp.asarray(packed['response'])       # [N, M]
    tau = cp.asarray(packed['tau'])                  # [N, M]
    act_id = cp.asarray(packed['activation_id'])    # [N, M]

    # Precompute exponential Euler constants.
    decay = cp.exp(-dt / tau)                       # [N, M]
    scale = 1.0 - decay                            # [N, M]

    # Input node slots: decay=1, scale=0 (values are overwritten anyway).
    decay[:, :num_inputs] = 1.0
    scale[:, :num_inputs] = 0.0

    # Transfer inputs to GPU.
    # Shape is either [num_steps, num_inputs] (broadcast) or [num_steps, N, num_inputs].
    inputs_gpu = cp.asarray(inputs_cpu.astype(np.float32))

    # Initialize state.
    u = cp.zeros((N, M), dtype=cp.float32)

    # Allocate output trajectory.
    out_start = num_inputs
    out_end = num_inputs + num_outputs
    trajectory = cp.zeros((N, num_steps, num_outputs), dtype=cp.float32)

    # Allocate scratch for activation kernel.
    s_buf = cp.empty((N, M), dtype=cp.float32)
    z_buf = cp.empty((N, M), dtype=cp.float32)

    # Get activation kernel.
    kernel = _get_activation_kernel()
    total = N * M
    block_size = 256
    grid_size = (total + block_size - 1) // block_size

    # Flatten arrays for kernel (contiguous [N*M]).
    bias_flat = bias.ravel()
    response_flat = response.ravel()
    act_id_flat = act_id.ravel()

    for step in range(num_steps):
        # Step 1: Set input node states.
        u[:, :num_inputs] = inputs_gpu[step]

        # Step 2: Batched matrix-vector multiply.
        # s = W @ u → [N, M] (treat u as [N, M, 1], squeeze result)
        cp.matmul(W, u[:, :, None], out=s_buf[:, :, None])

        # Step 3: Apply activation function via custom kernel.
        s_flat = s_buf.ravel()
        z_flat = z_buf.ravel()
        kernel((grid_size,), (block_size,),
               (s_flat, bias_flat, response_flat, act_id_flat, z_flat, total))

        # Step 4: Exponential Euler state update.
        # u = decay * u + scale * z
        cp.multiply(decay, u, out=u)
        u += scale * z_buf

        # Step 5: Re-clamp input nodes.
        u[:, :num_inputs] = inputs_gpu[step]

        # Step 6: Record output node states.
        trajectory[:, step, :] = u[:, out_start:out_end]

    return cp.asnumpy(trajectory)


def evaluate_iznn_batch(packed, inputs_cpu, dt, num_steps):
    """
    Run batched Izhikevich spiking network simulation on GPU.

    Parameters
    ----------
    packed : dict
        Output of pack_iznn_population(). NumPy arrays on CPU.
    inputs_cpu : ndarray [num_steps, num_inputs] or [num_steps, N, num_inputs]
        Precomputed input trajectory.
    dt : float
        Integration time step in milliseconds.
    num_steps : int
        Number of simulation steps.

    Returns
    -------
    trajectory : ndarray [N, num_steps, num_outputs] float32 on CPU
        Spike indicators (0.0 or 1.0) for output nodes at each step.
    """
    cp = _import_cupy()
    np = _import_numpy()

    N = packed['W'].shape[0]
    M = packed['max_nodes']
    num_inputs = packed['num_inputs']
    num_outputs = packed['num_outputs']

    # Transfer to GPU.
    W = cp.asarray(packed['W'])              # [N, M, M]
    bias = cp.asarray(packed['bias'])        # [N, M]
    a = cp.asarray(packed['a'])              # [N, M]
    b = cp.asarray(packed['b'])              # [N, M]
    c = cp.asarray(packed['c'])              # [N, M]
    d = cp.asarray(packed['d'])              # [N, M]
    node_mask = cp.asarray(packed['node_mask'])  # [N, M]

    # Transfer inputs to GPU.
    # Shape is either [num_steps, num_inputs] (broadcast) or [num_steps, N, num_inputs].
    inputs_gpu = cp.asarray(inputs_cpu.astype(np.float32))

    # Initialize state: v = c, u_recov = b * v, fired = 0.
    v = c.copy()
    u_recov = b * v
    fired = cp.zeros((N, M), dtype=cp.float32)

    # Source vector combines fired (for neurons) and external inputs.
    source = cp.zeros((N, M), dtype=cp.float32)

    out_start = num_inputs
    out_end = num_inputs + num_outputs
    trajectory = cp.zeros((N, num_steps, num_outputs), dtype=cp.float32)

    # Mask for non-input nodes (where integration happens).
    neuron_mask = node_mask.copy()
    neuron_mask[:, :num_inputs] = False
    neuron_mask_f = neuron_mask.astype(cp.float32)

    for step in range(num_steps):
        # Build source vector: fired for neuron slots, external input for input slots.
        source[:] = fired
        source[:, :num_inputs] = inputs_gpu[step]

        # Compute synaptic current: I = bias + W @ source
        I = bias + cp.matmul(W, source[:, :, None]).squeeze(-1)

        # Half-step voltage update (only for neuron nodes).
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u_recov + I
        v += 0.5 * dt * dv * neuron_mask_f

        dv = 0.04 * v * v + 5.0 * v + 140.0 - u_recov + I
        v += 0.5 * dt * dv * neuron_mask_f

        # Recovery variable update.
        u_recov += dt * a * (b * v - u_recov) * neuron_mask_f

        # Handle overflow (v becomes inf/nan): reset without spike.
        overflow = cp.isinf(v) | cp.isnan(v)
        v = cp.where(overflow, c, v)
        u_recov = cp.where(overflow, b * v, u_recov)

        # Spike detection and reset.
        spiked = v > 30.0
        fired = cp.where(spiked, 1.0, 0.0) * neuron_mask_f
        v = cp.where(spiked, c, v)
        u_recov = cp.where(spiked, u_recov + d, u_recov)

        # Record output: fired state of output nodes.
        trajectory[:, step, :] = fired[:, out_start:out_end]

    return cp.asnumpy(trajectory)
