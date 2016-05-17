import os

from neat import genome
from neat.config import Config


def check_simple(genome_type):
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))
    c1 = genome_type.create_unconnected(1, config)
    c1.connect_full()
    repr(c1)
    str(c1)

    # add two hidden nodes
    c1.add_hidden_nodes(2)
    repr(c1)
    str(c1)

    # apply some mutations
    c1.mutate_add_node()
    c1.mutate_add_connection()
    repr(c1)
    str(c1)


def test_recurrent():
    check_simple(genome.Genome)


def test_feed_forward():
    check_simple(genome.FFGenome)


def check_self_crossover(genome_type):
    # Check that self-crossover produces a genetically identical child (with a different ID).
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))
    c = genome_type.create_unconnected(1, config)
    c.connect_full()
    c.fitness = 0.0

    cnew = c.crossover(c, 2)
    assert cnew.ID != c.ID
    assert len(cnew.conn_genes) == len(c.conn_genes)
    for kold, vold in cnew.conn_genes.items():
        print(kold, vold)
        print(c.conn_genes)
        assert kold in c.conn_genes
        vnew = c.conn_genes[kold]
        assert vold.is_same_innov(vnew)

        assert vnew.weight == vold.weight
        assert vnew.in_node_id == vold.in_node_id
        assert vnew.out_node_id == vold.out_node_id
        assert vnew.enabled == vold.enabled

    assert len(cnew.node_genes) == len(c.node_genes)
    for kold, vold in cnew.node_genes.items():
        assert kold in c.node_genes
        vnew = c.node_genes[kold]

        assert vnew.ID == vold.ID
        assert vnew.type == vold.type
        assert vnew.bias == vold.bias
        assert vnew.response == vold.response
        assert vnew.activation_type == vold.activation_type


def test_recurrent_self_crossover():
    check_self_crossover(genome.Genome)


def test_feed_forward_self_crossover():
    check_self_crossover(genome.FFGenome)


def check_add_connection(genome_type, feed_forward):
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))
    config.input_nodes = 3
    config.output_nodes = 4
    config.hidden_nodes = 5
    config.feedforward = feed_forward
    N = config.input_nodes + config.hidden_nodes + config.output_nodes

    connections = {}
    for a in range(100):
        g = genome_type.create_unconnected(a, config)
        g.add_hidden_nodes(config.hidden_nodes)
        for b in range(1000):
            g.mutate_add_connection()
        for c in g.conn_genes.values():
            connections[c.key] = connections.get(c.key, 0) + 1

    # TODO: The connections should be returned to the caller and checked
    # against the constraints/assumptions particular to the network type.
    for i in range(N):
        values = []
        for j in range(N):
            values.append(connections.get((i, j), 0))
        print("{0:2d}: {1}".format(i, " ".join("{0:3d}".format(x) for x in values)))


def test_recurrent_add_connection():
    check_add_connection(genome.Genome, 0)


def test_feed_forward_add_connection():
    check_add_connection(genome.FFGenome, 1)