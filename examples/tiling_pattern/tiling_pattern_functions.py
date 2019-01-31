from gym_multi_robot import visualize
from gym_multi_robot.genome_serializer import GenomeSerializer


def output_winner(winner, config, net_filename='nn_winner', genome_filename='winner'):
    """This function outputs the current winner in graph and in pickle file."""

    node_names = {-1: 'hold', -2: 'on object', -3: '1_obstacle', -4: '1_tile', -5: '1_robot', -6: '2_obstacle',
                  -7: '2_tile', -8: '2_robot', -9: '3_obstacle', -10: '3_tile', -11: '3_robot', -12: '4_obstacle',
                  -13: '4_tile', -14: '4_robot', -15: '5_obstacle', -16: '5_tile', -17: '5_robot',
                  0: 'drive', 1: 'rotation', 2: 'pickup', 3: 'put down'}
    visualize.draw_state_machine(config, winner, node_names=node_names, filename=net_filename)

    GenomeSerializer.serialize(winner, genome_filename)

    print(winner)