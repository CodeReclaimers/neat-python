"""Directed graph algorithm implementations."""
from collections import defaultdict, deque

def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """
    assert not set(inputs).intersection(outputs)

    # Create a graph representation of the connections
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    for a, b in connections:
        graph[a].append(b)
        reverse_graph[b].append(a)

    # Perform a breadth-first search (BFS) from each input to find all reachable nodes
    reachable = set(inputs)
    queue = deque(inputs)

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    # Now, traverse from the outputs and find all nodes that are required to reach the outputs
    required = set(outputs)
    s = set(outputs)
    while True:
        # Find nodes not in s whose output is consumed by a node in s and is reachable from inputs
        t = set(a for (a, b) in connections if b in s and a not in s and a in reachable)

        if not t:
            break

        # Only add non-input nodes to the required set
        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)

    layers = []
    potential_input = set(inputs)
    while True:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in connections if a in potential_input and b not in potential_input)
        # Keep only the used nodes whose entire input set is contained in s.
        next_layer = set()
        for n in c:
            # select connections (a, b) where b == n
            connections_to_n = [(a, b) for (a, b) in connections if b == n and a in required]
            if n in required and all(a in potential_input for (a, b) in connections_to_n):
                next_layer.add(n)

        if not next_layer:
            break

        layers.append(next_layer)
        potential_input = potential_input.union(next_layer)

    return layers, required

