import unittest

from neat.state_machine_genes import TransitionGene
from neat.state_machine_genome import StateMachineGenome
from neat.state_machine_network import Condition
from tests.config_generation import init_config


class TestTransitionGene(unittest.TestCase):

    def test_distance(self):
        config = init_config()
        genome1 = TransitionGene((1, 2))
        genome2 = TransitionGene((1, 2))

        genome1.init_attributes(config)
        genome2.init_attributes(config)

        self.assertEqual(0, genome1.distance(genome2, config))

    def test_distance_1(self):
        config = init_config()
        genome1 = TransitionGene((1, 2))
        genome2 = TransitionGene((1, 2))

        genome1.init_attributes(config)
        genome2.init_attributes(config)

        genome1.conditions.append(Condition(1, Condition.random_operator(), 1))

        self.assertEqual(1, genome1.distance(genome2, config))

    def test_distance_2(self):
        config = init_config()
        genome1 = TransitionGene((1, 2))
        genome2 = TransitionGene((1, 2))

        genome1.init_attributes(config)
        genome2.init_attributes(config)

        genome2.conditions.append(Condition(1, Condition.random_operator(), 1))

        self.assertEqual(1, genome1.distance(genome2, config))

    def test_distance_3(self):
        config = init_config()
        genome1 = TransitionGene((1, 2))
        genome2 = TransitionGene((1, 2))

        genome1.init_attributes(config)
        genome2.init_attributes(config)

        genome2.conditions.append(Condition(1, Condition.random_operator(), 1))
        genome2.conditions.append(Condition(1, Condition.random_operator(), 1))
        genome2.conditions.append(Condition(1, Condition.random_operator(), 1))
        genome2.conditions.append(Condition(1, Condition.random_operator(), 1))

        self.assertEqual(4, genome1.distance(genome2, config))

class TestStateMachineGenome(unittest.TestCase):

    def test_single_state(self):
        genome = StateMachineGenome(1)
        config = init_config()
        genome.configure_new(config)

        self.assertEqual(len(genome.states), 1)
        self.assertEqual(len(genome.transitions), 0)

    def test_two_state_genome(self):
        genome = StateMachineGenome(1)
        config = init_config()
        genome.configure_new(config)
        genome.mutate_add_state(config)

        self.assertEqual(len(genome.states), 2)
        self.assertEqual(len(genome.transitions), 2)
        self.assertIn((0, 1), genome.transitions)
        self.assertIn((1, 0), genome.transitions)

    def test_add_transition_to_existing(self):
        genome = StateMachineGenome(1)
        config = init_config()
        genome.configure_new(config)
        genome.mutate_add_state(config)

        genome.mutate_add_transition(config)

        self.assertEqual(2, len(genome.states))
        self.assertEqual(2, len(genome.transitions))

    def test_add_transition_transition_free_state_machine(self):
        genome = StateMachineGenome(1)
        config = init_config()
        genome.configure_new(config)
        state = genome.create_state(config, 1)
        genome.states[1] = state

        self.assertEqual(len(genome.states), 2)
        self.assertEqual(len(genome.transitions), 0)

        try:
            genome.mutate_add_transition(config)
            self.assertEqual(1, len(genome.transitions))
        except AssertionError:
            print("Same state, all ok.")

    def test_multiple_initial_states(self):
        genome = StateMachineGenome(1)
        config = init_config()
        config.num_initial_states = 4
        genome.configure_new(config)

        self.assertEqual(len(genome.states), 4)
        self.assertEqual(len(genome.transitions), 0)

    def test_state_difference_distance(self):
        genome = StateMachineGenome(1)
        genome2 = StateMachineGenome(2)
        config = init_config()
        config.state_difference_coefficient = 0
        config.compatibility_disjoint_coefficient = 1
        config.num_initial_states = config.node_indexer = 4
        genome.configure_new(config)
        genome2.configure_new(config)
        genome2.mutate_add_state(config)

        self.assertEqual(1, genome.distance(genome2, config))


if __name__ == '__main__':
    unittest.main()
