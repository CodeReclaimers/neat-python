import math
import os
import random

import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene


# def test_concurrent_nn():
#     """This is a stripped-down copy of the `memory` example."""
#
#     # num_tests is the number of random examples each network is tested against.
#     num_tests = 16
#     # N is the length of the test sequence.
#     N = 4
#
#     def eval_fitness(genomes):
#         for g in genomes:
#             net = nn.create_recurrent_phenotype(g)
#
#             error = 0.0
#             for _ in range(num_tests):
#                 # Create a random sequence, and feed it to the network with the
#                 # second input set to zero.
#                 seq = [random.choice((0, 1)) for _ in range(N)]
#                 net.reset()
#                 for s in seq:
#                     inputs = [s, 0]
#                     net.activate(inputs)
#
#                 # Set the second input to one, and get the network output.
#                 for s in seq:
#                     inputs = [0, 1]
#                     output = net.activate(inputs)
#
#                     error += (output[0] - s) ** 2
#
#             g.fitness = -(error / (N * num_tests)) ** 0.5
#
#     # Demonstration of how to add your own custom activation function.
#     def sinc(x):
#         return 1.0 if x == 0 else math.sin(x) / x
#
#     # This sinc function will be available if my_sinc_function is included in the
#     # config file activation_functions option under the pheotype section.
#     # Note that sinc is not necessarily useful for this example, it was chosen
#     # arbitrarily just to demonstrate adding a custom activation function.
#     activation_functions.add('my_sinc_function', sinc)
#
#     local_dir = os.path.dirname(__file__)
#     pop = population.Population(os.path.join(local_dir, 'recurrent_config'))
#     pop.run(eval_fitness, 10)
#
#     # Visualize the winner network and plot/log statistics.
#     # visualize.draw_net(winner, view=True, filename="nn_winner.gv")
#     # visualize.draw_net(winner, view=True, filename="nn_winner-enabled.gv", show_disabled=False)
#     # visualize.draw_net(winner, view=True, filename="nn_winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)
#     # visualize.plot_stats(pop.statistics)
#     # visualize.plot_species(pop.statistics)
#     statistics.save_stats(pop.statistics)
#     statistics.save_species_count(pop.statistics)
#     statistics.save_species_fitness(pop.statistics)
#
#     winner = pop.statistics.best_genome()
#     repr(winner)
#     str(winner)
#     for g in winner.node_genes:
#         repr(g)
#         str(g)
#     for g in winner.conn_genes:
#         repr(g)
#         str(g)


def test_orphaned_node_network():
    """Test that networks with orphaned nodes (no incoming connections) work correctly."""
    # Load configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Create a genome with an orphaned hidden node
    genome = neat.DefaultGenome(1)
    genome.fitness = None
    
    # Manually create nodes:
    # - Output node 0
    # - Hidden node 1 (orphaned - no incoming connections)
    node_0 = DefaultNodeGene(0)
    node_0.bias = 0.5
    node_0.response = 1.0
    node_0.activation = 'sigmoid'
    node_0.aggregation = 'sum'
    genome.nodes[0] = node_0
    
    node_1 = DefaultNodeGene(1)
    node_1.bias = 2.0  # This bias value should affect the output
    node_1.response = 1.0
    node_1.activation = 'sigmoid'
    node_1.aggregation = 'sum'
    genome.nodes[1] = node_1
    
    # Manually create connections:
    # - Node 1 (orphaned) connects to output node 0
    # - Note: node 1 has no incoming connections (it's orphaned)
    conn_key = (1, 0)
    conn = DefaultConnectionGene(conn_key, innovation=0)  # Innovation number required
    conn.weight = 1.0
    conn.enabled = True
    genome.connections[conn_key] = conn
    
    # Create the feed-forward network
    # This should not crash despite node 1 being orphaned
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Activate the network
    # The orphaned node 1 should contribute its activation(bias) to the output
    output = net.activate([0.0, 0.0])  # Two inputs (as per test_configuration)
    
    # Verify the network produces output
    assert len(output) == 1
    assert output[0] is not None
    
    # Verify the output is affected by the orphaned node's bias
    # NOTE: neat-python's sigmoid_activation multiplies input by 5.0 before applying sigmoid!
    # Node 1 with bias=2.0, no inputs -> sigmoid(5.0 * (2.0 + 1.0 * 0)) = sigmoid(10.0) ≈ 0.9999
    # Node 1 output (0.9999) * weight (1.0) goes to node 0
    # Node 0: sigmoid(5.0 * (0.5 + 1.0 * 0.9999)) = sigmoid(5.0 * 1.4999) = sigmoid(7.5) ≈ 0.9994
    
    # Calculate using neat-python's sigmoid_activation (with 5.0 scaling)
    def neat_sigmoid(z):
        z = max(-60.0, min(60.0, 5.0 * z))
        return 1.0 / (1.0 + math.exp(-z))
    
    expected_node_1_output = neat_sigmoid(2.0)  # bias only, no inputs
    expected_output = neat_sigmoid(0.5 + expected_node_1_output)  # bias + weighted input from node 1
    
    assert abs(output[0] - expected_output) < 0.001, f"Expected {expected_output:.6f}, got {output[0]:.6f}"


if __name__ == '__main__':
    test_orphaned_node_network()
