import os
from neat import genes
from neat import genome
from neat.config import Config


def test_recurrent():
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))
    c1 = genome.Genome.create_fully_connected(config, genes.NodeGene, genes.ConnectionGene)

    # add two hidden nodes
    #c1.add_hidden_nodes(2)

    # apply some mutations
    c1._mutate_add_node()
    c1._mutate_add_connection()


def test_feed_forward():
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))
    c2 = genome.FFGenome.create_fully_connected(config, genes.NodeGene, genes.ConnectionGene)

    # add two hidden nodes
    #c2.add_hidden_nodes(2)

    # apply some mutations
    c2._mutate_add_node()
    c2._mutate_add_connection()
