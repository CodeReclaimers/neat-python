import pickle

from gym_multi_robot import visualize

from neat import Config, DefaultSpeciesSet, DefaultReproduction, DefaultStagnation
from neat.state_machine_genome import StateMachineGenome


def main():
    config = Config(StateMachineGenome, DefaultReproduction,
                         DefaultSpeciesSet, DefaultStagnation,
                    'config-state_machine')

    with open('winner_static.pickle', 'rb') as pickle_file:
        thing = pickle.load(pickle_file)

    thing.transitions[(1, 2)] = thing.create_transition(config.genome_config, 0, 1)

    visualize.draw_state_machine(config, thing, filename='test')


if __name__ == '__main__':
    main()