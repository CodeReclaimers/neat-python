""" Directed graph algorithm implementations. """


def creates_cycle(connections, test):
    """
    Returns true if the addition of the "test" connection would create a cycle,
    assuming that no cycle already exists in the graph represented by "connections".
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
    '''
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a list of layers, with each layer consisting of a set of identifiers.
    '''

    required = set(outputs)
    S = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in S.
        T = set(a for (a, b) in connections if b in S and a not in S)

        if not T:
            break

        layer_nodes = set(x for x in T if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        S = S.union(T)

    return required


def feed_forward_layers(inputs, outputs, connections):
    '''
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    '''

    required = required_for_output(inputs, outputs, connections)

    layers = []
    S = set(inputs)
    while 1:
        # Find candidate nodes C for the next layer.  These nodes should connect
        # a node in S to a node not in S.
        C = set(b for (a, b) in connections if a in S and b not in S)
        # Keep only the used nodes whose entire input set is contained in S.
        T = set()
        for n in C:
            if n in required and all(a in S for (a, b) in connections if b == n):
                T.add(n)

        if not T:
            break

        layers.append(T)
        S = S.union(T)

    return layers


