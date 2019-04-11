import pickle
import unittest

from neat.state_machine_genome_partially_fixed import StateMachineGenomeFixed
from tests.config_generation import init_fixed_genome_config


class TestStateMachineGenome(unittest.TestCase):

    def test_fixed_layout(self):
        reference_genome = pickle.load(open('test_genome.pickle', "rb"))

        genome = StateMachineGenomeFixed(1)
        config = init_fixed_genome_config()
        config.fixed_section = 'layout'
        genome.configure_new(config)

        self.assertEqual(len(reference_genome.transitions), len(genome.transitions))
        self.assertEqual(len(reference_genome.states), len(genome.states))

    def test_fixed_states(self):
        reference_genome = pickle.load(open('test_genome.pickle', "rb"))

        genome = StateMachineGenomeFixed(1)
        config = init_fixed_genome_config()
        # Default fixed section are the states.

        genome.configure_new(config)

        # Check lengths.
        self.assertEqual(0, len(genome.transitions))
        self.assertEqual(2, len(genome.states))

        # Check state availability.
        self.assertIn(0, genome.states)
        self.assertIn(1, genome.states)

        # Check equality of the state layouts.
        self.assertEqual(reference_genome.states[0].biases, genome.states[0].biases)
        self.assertEqual(reference_genome.states[0].weights, genome.states[0].weights)
        self.assertEqual(reference_genome.states[1].biases, genome.states[1].biases)
        self.assertEqual(reference_genome.states[1].weights, genome.states[1].weights)

    def test_fixed_transitions(self):
        reference_genome = pickle.load(open('test_genome.pickle', "rb"))

        genome = StateMachineGenomeFixed(1)
        config = init_fixed_genome_config()
        config.fixed_section = 'transitions'

        genome.configure_new(config)

        # Check lengths.
        self.assertEqual(1, len(genome.transitions))
        self.assertEqual(2, len(genome.states))

        # Check state availability.
        self.assertIn(0, genome.states)
        self.assertIn(1, genome.states)

        # Check transition availability
        self.assertIn((0, 1), genome.transitions)
        self.assertEqual(reference_genome.transitions[(0, 1)].conditions,
                         genome.transitions[(0, 1)].conditions)
        # Manually checked that states do randomly initialise.

    def test_state_fixed_with_mutation(self):

        genome = StateMachineGenomeFixed(1)
        config = init_fixed_genome_config()
        genome.configure_new(config)

        reference_genome = pickle.load(open('test_genome.pickle', "rb"))

        # Do 100 mutations to check whether it always is the same.
        for i in range(100):
            genome.mutate(config)

            self.assertEqual(2, len(genome.states))

            # Check equality of the state layouts.
            self.assertEqual(reference_genome.states[0].biases, genome.states[0].biases)
            self.assertEqual(reference_genome.states[0].weights, genome.states[0].weights)
            self.assertEqual(reference_genome.states[1].biases, genome.states[1].biases)
            self.assertEqual(reference_genome.states[1].weights, genome.states[1].weights)

    def test_transition_fixed_with_mutation(self):
        genome = StateMachineGenomeFixed(1)
        config = init_fixed_genome_config()
        config.fixed_section = 'transitions'
        genome.configure_new(config)

        reference_genome = pickle.load(open('test_genome.pickle', "rb"))

        # Do 100 mutations to check whether it always is the same.
        for _ in range(100):
            genome.mutate(config)

            self.assertEqual(2, len(genome.states))
            self.assertEqual(1, len(genome.transitions))

            # Check transition availability
            self.assertIn((0, 1), genome.transitions)
            self.assertEqual(reference_genome.transitions[(0, 1)].conditions,
                             genome.transitions[(0, 1)].conditions)

    def test_hard_copy_transition(self):
        """ Ensure that a hard copy of the transitions is made, so they can be individually changed. """
        reference_genome = pickle.load(open('test_genome.pickle', "rb"))

        genome = StateMachineGenomeFixed(1)
        config = init_fixed_genome_config()
        config.fixed_section = 'transitions'

        genome.configure_new(config)

        # Check lengths.
        self.assertEqual(1, len(genome.transitions))

        # Check transition availability
        self.assertIn((0, 1), genome.transitions)
        genome.transitions[(0, 1)].conditions.append((1, 2, 3))
        self.assertNotEqual(reference_genome.transitions[(0, 1)].conditions,
                            genome.transitions[(0, 1)].conditions)

    def test_hard_copy_state(self):
        """ Ensure that a hard copy of the states can be changed so they can be individually changed. """
        reference_genome = pickle.load(open('test_genome.pickle', "rb"))

        genome = StateMachineGenomeFixed(1)
        config = init_fixed_genome_config()
        config.fixed_section = 'states'

        genome.configure_new(config)
        genome.states[0].biases[0] = -1
        genome.states[0].weights[0][0] = -1

        self.assertNotEqual(reference_genome.states[0].biases, genome.states[0].biases)
        self.assertNotEqual(reference_genome.states[0].weights, genome.states[0].weights)
