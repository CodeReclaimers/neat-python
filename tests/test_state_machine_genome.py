import unittest

from neat.state_machine_genome import StateMachineGenome
from tests.config_generation import init_config


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


if __name__ == '__main__':
    unittest.main()
