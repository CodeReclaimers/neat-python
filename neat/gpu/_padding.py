"""
Genome-to-tensor conversion for GPU evaluation.

Converts variable-topology neat-python genomes into fixed-size padded NumPy
arrays suitable for batched GPU evaluation. This module is pure Python/NumPy
and does not import CuPy.

Index layout within the M-slot node vector:
    [0 .. num_inputs-1]                          → input pins
    [num_inputs .. num_inputs+num_outputs-1]      → output nodes
    [num_inputs+num_outputs .. max_nodes-1]        → hidden nodes (per-genome)
"""

from neat.gpu import _import_numpy
from neat.graphs import required_for_output

# Activation function name → integer ID.
# These must match the dispatch in _cupy_backend.py.
ACTIVATION_IDS = {
    'sigmoid': 0,
    'tanh': 1,
    'relu': 2,
    'identity': 3,
    'clamped': 4,
    'elu': 5,
    'softplus': 6,
    'sin': 7,
    'gauss': 8,
    'abs': 9,
    'square': 10,
}

_UNSUPPORTED_AGGREGATIONS = frozenset([
    'product', 'max', 'min', 'maxabs', 'median', 'mean',
])


def _build_node_key_map(genome, config, required_nodes):
    """
    Build a mapping from neat-python node keys to dense indices.

    Returns (key_map, num_nodes) where key_map is {node_key: index} and
    num_nodes is the total number of slots used by this genome (including
    input pins).
    """
    genome_config = config.genome_config
    key_map = {}

    # Input pins first.
    for idx, k in enumerate(genome_config.input_keys):
        key_map[k] = idx

    num_inputs = len(genome_config.input_keys)

    # Output nodes next.
    for idx, k in enumerate(genome_config.output_keys):
        key_map[k] = num_inputs + idx

    num_outputs = len(genome_config.output_keys)
    hidden_start = num_inputs + num_outputs

    # Hidden nodes: only those in the required set.
    hidden_idx = 0
    for k in sorted(required_nodes):
        if k not in key_map:
            key_map[k] = hidden_start + hidden_idx
            hidden_idx += 1

    num_nodes = hidden_start + hidden_idx
    return key_map, num_nodes


def pack_ctrnn_population(genomes, config):
    """
    Convert a list of (genome_id, genome) pairs into padded NumPy arrays
    for GPU CTRNN evaluation.

    Parameters
    ----------
    genomes : list of (int, genome) tuples
        The population to pack.
    config : neat.Config
        The NEAT configuration object.

    Returns
    -------
    dict with keys:
        W : ndarray [N, M, M] float32 — weight matrices
        bias : ndarray [N, M] float32
        response : ndarray [N, M] float32
        tau : ndarray [N, M] float32
        activation_id : ndarray [N, M] int32
        node_mask : ndarray [N, M] bool
        num_inputs : int
        num_outputs : int
        max_nodes : int
        node_key_maps : list of dict — per-genome {node_key: dense_index}
    """
    np = _import_numpy()
    genome_config = config.genome_config
    num_inputs = len(genome_config.input_keys)
    num_outputs = len(genome_config.output_keys)
    N = len(genomes)

    # First pass: determine max_nodes across population.
    per_genome_info = []
    max_nodes = num_inputs + num_outputs

    for genome_id, genome in genomes:
        required = required_for_output(
            genome_config.input_keys, genome_config.output_keys,
            genome.connections)
        key_map, num_nodes = _build_node_key_map(genome, config, required)
        per_genome_info.append((genome_id, genome, required, key_map, num_nodes))
        if num_nodes > max_nodes:
            max_nodes = num_nodes

    M = max_nodes

    # Allocate arrays.
    W = np.zeros((N, M, M), dtype=np.float32)
    bias = np.zeros((N, M), dtype=np.float32)
    response = np.ones((N, M), dtype=np.float32)  # default 1.0
    tau = np.ones((N, M), dtype=np.float32)  # default 1.0 (won't matter for masked-out nodes)
    activation_id = np.zeros((N, M), dtype=np.int32)  # default 0 = sigmoid
    node_mask = np.zeros((N, M), dtype=bool)

    # Input and output slots are active for all genomes.
    node_mask[:, :num_inputs] = True
    node_mask[:, num_inputs:num_inputs + num_outputs] = True

    node_key_maps = []

    # Second pass: fill arrays.
    for g_idx, (genome_id, genome, required, key_map, num_nodes) in enumerate(per_genome_info):
        node_key_maps.append(key_map)

        # Mark hidden nodes as active.
        for node_key in required:
            dense_idx = key_map[node_key]
            if dense_idx >= num_inputs:  # skip input pins
                node_mask[g_idx, dense_idx] = True

        # Fill node parameters for required (non-input) nodes.
        for node_key in required:
            dense_idx = key_map[node_key]
            node = genome.nodes[node_key]

            bias[g_idx, dense_idx] = node.bias
            response[g_idx, dense_idx] = node.response
            tau[g_idx, dense_idx] = node.time_constant

            # Validate and map activation function.
            act_name = node.activation
            if act_name not in ACTIVATION_IDS:
                raise ValueError(
                    f"Genome {genome_id}, node {node_key}: activation function "
                    f"'{act_name}' is not supported on GPU. Supported: "
                    f"{sorted(ACTIVATION_IDS.keys())}")
            activation_id[g_idx, dense_idx] = ACTIVATION_IDS[act_name]

            # Validate aggregation function.
            agg_name = node.aggregation
            if agg_name != 'sum':
                raise ValueError(
                    f"Genome {genome_id}, node {node_key}: aggregation function "
                    f"'{agg_name}' is not supported on GPU. Only 'sum' aggregation "
                    f"is supported (required for batched matrix-vector multiply).")

        # Fill weight matrix from enabled connections.
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            src_key, dst_key = cg.key
            # Only include connections where both endpoints are in the key map.
            if src_key not in key_map or dst_key not in key_map:
                continue
            # dst must be a required node (non-input).
            if dst_key not in required:
                continue

            src_idx = key_map[src_key]
            dst_idx = key_map[dst_key]
            W[g_idx, dst_idx, src_idx] = cg.weight

    return {
        'W': W,
        'bias': bias,
        'response': response,
        'tau': tau,
        'activation_id': activation_id,
        'node_mask': node_mask,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'max_nodes': M,
        'node_key_maps': node_key_maps,
    }


def pack_iznn_population(genomes, config):
    """
    Convert a list of (genome_id, genome) pairs into padded NumPy arrays
    for GPU Izhikevich spiking network evaluation.

    Parameters
    ----------
    genomes : list of (int, genome) tuples
        The population to pack.
    config : neat.Config
        The NEAT configuration object.

    Returns
    -------
    dict with keys:
        W : ndarray [N, M, M] float32 — weight matrices
        bias : ndarray [N, M] float32
        a : ndarray [N, M] float32
        b : ndarray [N, M] float32
        c : ndarray [N, M] float32
        d : ndarray [N, M] float32
        node_mask : ndarray [N, M] bool
        num_inputs : int
        num_outputs : int
        max_nodes : int
        node_key_maps : list of dict — per-genome {node_key: dense_index}
    """
    np = _import_numpy()
    genome_config = config.genome_config
    num_inputs = len(genome_config.input_keys)
    num_outputs = len(genome_config.output_keys)
    N = len(genomes)

    # First pass: determine max_nodes.
    per_genome_info = []
    max_nodes = num_inputs + num_outputs

    for genome_id, genome in genomes:
        required = required_for_output(
            genome_config.input_keys, genome_config.output_keys,
            genome.connections)
        key_map, num_nodes = _build_node_key_map(genome, config, required)
        per_genome_info.append((genome_id, genome, required, key_map, num_nodes))
        if num_nodes > max_nodes:
            max_nodes = num_nodes

    M = max_nodes

    # Allocate arrays.
    W = np.zeros((N, M, M), dtype=np.float32)
    bias_arr = np.zeros((N, M), dtype=np.float32)
    a_arr = np.zeros((N, M), dtype=np.float32)
    b_arr = np.zeros((N, M), dtype=np.float32)
    c_arr = np.full((N, M), -65.0, dtype=np.float32)  # default reset voltage
    d_arr = np.zeros((N, M), dtype=np.float32)
    node_mask = np.zeros((N, M), dtype=bool)

    node_mask[:, :num_inputs] = True
    node_mask[:, num_inputs:num_inputs + num_outputs] = True

    node_key_maps = []

    # Second pass: fill arrays.
    for g_idx, (genome_id, genome, required, key_map, num_nodes) in enumerate(per_genome_info):
        node_key_maps.append(key_map)

        for node_key in required:
            dense_idx = key_map[node_key]
            if dense_idx >= num_inputs:
                node_mask[g_idx, dense_idx] = True

        # Fill node parameters.
        for node_key in required:
            dense_idx = key_map[node_key]
            node = genome.nodes[node_key]

            bias_arr[g_idx, dense_idx] = node.bias
            a_arr[g_idx, dense_idx] = node.a
            b_arr[g_idx, dense_idx] = node.b
            c_arr[g_idx, dense_idx] = node.c
            d_arr[g_idx, dense_idx] = node.d

        # Fill weight matrix.
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            src_key, dst_key = cg.key
            if src_key not in key_map or dst_key not in key_map:
                continue
            if dst_key not in required:
                continue

            src_idx = key_map[src_key]
            dst_idx = key_map[dst_key]
            W[g_idx, dst_idx, src_idx] = cg.weight

    return {
        'W': W,
        'bias': bias_arr,
        'a': a_arr,
        'b': b_arr,
        'c': c_arr,
        'd': d_arr,
        'node_mask': node_mask,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'max_nodes': M,
        'node_key_maps': node_key_maps,
    }
