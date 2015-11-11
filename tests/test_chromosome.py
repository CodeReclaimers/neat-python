from neat import genome
from neat import chromosome
from neat.config import Config


def test_recurrent():
    config = Config('test_configuration')
    c1 = chromosome.Chromosome.create_fully_connected(config, genome.NodeGene, genome.ConnectionGene)

    # add two hidden nodes
    #c1.add_hidden_nodes(2)

    # apply some mutations
    c1._mutate_add_node()
    c1._mutate_add_connection()


def test_feed_forward():
    config = Config('test_configuration')
    c2 = chromosome.FFChromosome.create_fully_connected(config, genome.NodeGene, genome.ConnectionGene)

    # add two hidden nodes
    #c2.add_hidden_nodes(2)

    # apply some mutations
    c2._mutate_add_node()
    c2._mutate_add_connection()
