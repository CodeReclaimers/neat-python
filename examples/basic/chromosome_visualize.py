from neat import visualize, genome
from neat import chromosome
from neat.config import Config


# Example
# define some attributes
node_gene_type = genome.NodeGene  # standard neuron model
conn_gene_type = genome.ConnectionGene  # and connection link
Config.nn_activation = 'exp'  # activation function
Config.weight_stdev = 0.9  # weights distribution

Config.input_nodes = 2  # number of inputs
Config.output_nodes = 1  # number of outputs

# creates a chromosome for recurrent networks
# c1 = Chromosome.create_fully_connected()

# creates a chromosome for feedforward networks
chromosome.node_gene_type = genome.NodeGene

c2 = chromosome.FFChromosome.create_fully_connected()
# add two hidden nodes
c2.add_hidden_nodes(2)
# apply some mutations
# c2._mutate_add_node()
# c2._mutate_add_connection()

# check the result
# visualize.draw_net(c1) # for recurrent nets
visualize.draw_net(c2, view=True)  # for feedforward nets
# print the chromosome
print c2
