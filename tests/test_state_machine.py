import operator
import unittest

from neat.state_machine_genome import StateMachineGenome
from neat.state_machine_network import StateMachineNetwork, State, Transition, Condition
from tests.config_generation import init_config


class TestStateMachine(unittest.TestCase):

    def test_init(self):
        sm = StateMachineNetwork([], [])
        self.assertEqual(len(sm.transitions), 0)
        self.assertEqual(len(sm.states), 0)

    def test_add_state(self):
        sm = StateMachineNetwork([State(1)], [])
        self.assertEqual(len(sm.states), 1)
        self.assertEqual(sm.states[1].id, 1)

    def test_add_state_twice(self):
        with self.assertRaises(ValueError) as context:
            StateMachineNetwork([State(1), State(1)], [])

        self.assertTrue('State included twice' in str(context.exception))

    def test_transition(self):
        sm = StateMachineNetwork([State(1), State(2)], [Transition(1, 2)])
        self.assertEqual(len(sm.transitions), 2)
        self.assertEqual(len(sm.transitions[1]), 1)

    def test_transition_multiple(self):
        sm = StateMachineNetwork([State(1), State(2)], [Transition(1, 2), Transition(2, 1)])
        self.assertEqual(len(sm.transitions), 2)
        self.assertEqual(len(sm.transitions[1]), 1)
        self.assertEqual(len(sm.transitions[2]), 1)

    def test_transition_multiple_begin_state(self):
        sm = StateMachineNetwork([State(1), State(2), State(3)], [Transition(1, 2), Transition(1, 3)])
        self.assertEqual(len(sm.transitions), 3)
        self.assertEqual(len(sm.transitions[1]), 2)

    def test_invalid_transition(self):
        self.assertRaises(ValueError, StateMachineNetwork, [State(1)], [Transition(4, 2)])

    def test_execution_no_transitions(self):
        sm = StateMachineNetwork([State(1)], [])
        sm.states[1].set_biases([1, 2])
        sm.states[1].set_weights([[1, 2], [3, 4]])

        next_state, output = sm.activate(1, [5, 10])
        self.assertEqual(next_state, 1)
        self.assertEqual(output, [26, 57])

    def test_execution_false_transition(self):
        sm = StateMachineNetwork([State(1), State(2)], [Transition(1, 2)])
        sm.states[1].set_biases([1, 2])
        sm.states[1].set_weights([[1, 2], [3, 4]])
        sm.transitions[1][0].add_condition(Condition(0, operator.eq, 4))

        next_state, output = sm.activate(1, [5, 10])
        self.assertEqual(next_state, 1)
        self.assertEqual(output, [26, 57])

    def test_execution_true_transition(self):
        sm = StateMachineNetwork([State(1), State(2)], [Transition(1, 2)])
        sm.states[1].set_biases([1, 2])
        sm.states[1].set_weights([[1, 2], [3, 4]])
        sm.transitions[1][0].add_condition(Condition(0, operator.eq, 5))

        next_state, output = sm.activate(1, [5, 10])
        self.assertEqual(next_state, 2)
        self.assertEqual(output, [26, 57])

    def test_execution_multiple_conditions_all_false(self):
        sm = StateMachineNetwork([State(1), State(2)], [Transition(1, 2)])
        sm.states[1].set_biases([1, 2])
        sm.states[1].set_weights([[1, 2], [3, 4]])
        sm.transitions[1][0].add_condition(Condition(0, operator.eq, 4))
        sm.transitions[1][0].add_condition(Condition(1, operator.gt, 10))

        next_state, output = sm.activate(1, [5, 10])
        self.assertEqual(next_state, 1)
        self.assertEqual(output, [26, 57])

    def test_execution_multiple_transitions_all_true(self):
        sm = StateMachineNetwork([State(1), State(2), State(3)], [Transition(1, 2), Transition(1, 2)])
        sm.states[1].set_biases([1, 2])
        sm.states[1].set_weights([[1, 2], [3, 4]])
        sm.transitions[1][0].add_condition(Condition(0, operator.eq, 5))
        sm.transitions[1][1].add_condition(Condition(1, operator.gt, 5))

        next_state, output = sm.activate(1, [5, 10])
        self.assertIn(next_state, [2, 3])
        self.assertEqual(output, [26, 57])

    def test_execution_multiple_transitions_all_false(self):
        sm = StateMachineNetwork([State(1), State(2), State(3)], [Transition(1, 2), Transition(1, 2)])
        sm.states[1].set_biases([1, 2])
        sm.states[1].set_weights([[1, 2], [3, 4]])
        sm.transitions[1][0].add_condition(Condition(0, operator.eq, 4))
        sm.transitions[1][1].add_condition(Condition(1, operator.gt, 10))

        next_state, output = sm.activate(1, [5, 10])
        self.assertEqual(1, next_state)
        self.assertEqual(output, [26, 57])

    def test_invalid_state(self):
        sm = StateMachineNetwork([State(1)], [])
        self.assertRaises(KeyError, sm.activate, 3, [5, 10])

    def test_empty_sm_creation(self):
        genome = StateMachineGenome(1)
        smn = StateMachineNetwork.create(genome, init_config())

        self.assertEqual(len(smn.states), 0)
        self.assertEqual(len(smn.transitions), 0)

    def test_one_state(self):
        genome = StateMachineGenome(1)
        config = init_config()
        genome.configure_new(config)
        smn = StateMachineNetwork.create(genome, config)

        self.assertEqual(len(smn.states), 1)
        self.assertEqual(len(smn.transitions), 1)

    def test_two_states(self):
        genome = StateMachineGenome(1)
        config = init_config()
        genome.configure_new(config)
        genome.mutate_add_state(config)
        smn = StateMachineNetwork.create(genome, config)

        self.assertEqual(len(smn.states), 2)
        self.assertEqual(len(smn.transitions), 2)
        self.assertIsNotNone(smn.states[0])
        self.assertIsNotNone(smn.states[1])
        self.assertEqual(len(smn.transitions[0]), 1)
        self.assertEqual(len(smn.transitions[1]), 1)

    def test_create_transition(self):
        genome = StateMachineGenome(1)
        config = init_config()
        genome.configure_new(config)
        genome.mutate_add_state(config)
        genome.mutate_add_transition(config)
        smn = StateMachineNetwork.create(genome, config)

        self.assertEqual(len(smn.states), 2)
        self.assertEqual(len(smn.transitions), 2)
        self.assertEqual(2, len(smn.transitions[1]) + len(smn.transitions[0]))


class TestState(unittest.TestCase):

    def test_init(self):
        state = State(42)
        self.assertEqual(state.id, 42)
        self.assertIsNone(state.biases)
        self.assertIsNone(state.weights)

    def test_not_without_biases(self):
        state = State(42)
        self.assertRaises(AssertionError, state.activate, [5, 10])

    def test_not_without_weights(self):
        state = State(42)
        state.set_biases([2, 3])
        self.assertRaises(AssertionError, state.activate, [5, 10])

    def test_to_long_bias_length(self):
        state = State(42)
        state.set_biases([2, 3, 4])
        state.set_weights([[2, 3], [4, 5]])
        self.assertRaises(AssertionError, state.activate, [5, 10])

    def test_too_short_bias_length(self):
        state = State(42)
        state.set_biases([2])
        state.set_weights([[2, 3], [4, 5]])
        self.assertRaises(AssertionError, state.activate, [5, 10])

    def test_to_many_weight_length(self):
        state = State(42)
        state.set_biases([2, 2])
        state.set_weights([[2, 3, 3], [3, 4, 5]])
        self.assertRaises(AssertionError, state.activate, [5, 10])

    def test_to_little_weight_length(self):
        state = State(42)
        state.set_biases([2, 3])
        state.set_weights([[2], [2]])
        self.assertRaises(AssertionError, state.activate, [5, 2])

    def test_2_in_1_out(self):
        state = State(42)
        state.set_biases([0])
        state.set_weights([[2, 3]])
        self.assertListEqual(state.activate([5, 10]), [40])

    def test_2_in_2_out(self):
        state = State(42)
        state.set_biases([0, 1])
        state.set_weights([[2, 3], [4, 5]])
        self.assertListEqual(state.activate([5, 10]), [40, 71])

    def test_2_in_3_out(self):
        state = State(42)
        state.set_biases([0, 1, 2])
        state.set_weights([[2, 3], [4, 5], [1, 2]])
        self.assertListEqual(state.activate([5, 10]), [40, 71, 27])

    def test_3_in_2_out(self):
        state = State(42)
        state.set_biases([0, 1])
        state.set_weights([[2, 3, 1], [4, 5, 1]])
        self.assertListEqual(state.activate([5, 10, 20]), [60, 91])

    def test_1_in_2_out(self):
        state = State(42)
        state.set_biases([0, 1])
        state.set_weights([[2], [4]])
        self.assertListEqual(state.activate([5]), [10, 21])

    def test_1_in_1_out(self):
        state = State(42)
        state.set_biases([1])
        state.set_weights([[2]])
        self.assertListEqual(state.activate([5]), [11])


class TestTransition(unittest.TestCase):

    def test_init(self):
        transition = Transition(1, 2)
        self.assertEqual(transition.begin_state_id, 1)
        self.assertEqual(transition.end_state_id, 2)

    def test_no_codition(self):
        transition = Transition(1, 2)
        self.assertTrue(transition.check_transition([0, 0, 0]))

    def test_one_condition_true(self):
        transition = Transition(1, 2)
        transition.add_condition(Condition(2, operator.eq, 5))
        self.assertTrue(transition.check_transition([0, 0, 5]))

    def test_one_condition_false(self):
        transition = Transition(1, 2)
        transition.add_condition(Condition(2, operator.eq, 5))
        self.assertFalse(transition.check_transition([0, 0, 4]))

    def test_two_conditions_false(self):
        transition = Transition(1, 2)
        transition.add_condition(Condition(1, operator.eq, 5))
        transition.add_condition(Condition(2, operator.eq, 4))
        self.assertFalse(transition.check_transition([0, 0, 4]))

    def test_two_conditions_true(self):
        transition = Transition(1, 2)
        transition.add_condition(Condition(1, operator.lt, 5))
        transition.add_condition(Condition(2, operator.eq, 4))
        self.assertTrue(transition.check_transition([0, 0, 4]))

    def test_many_conditions_false(self):
        transition = Transition(1, 2)
        transition.add_condition(Condition(1, operator.lt, 5))
        transition.add_condition(Condition(2, operator.eq, 4))
        transition.add_condition(Condition(2, operator.lt, 5))
        transition.add_condition(Condition(0, operator.eq, 4))
        transition.add_condition(Condition(1, operator.eq, 0))
        transition.add_condition(Condition(0, operator.lt, 4))
        self.assertFalse(transition.check_transition([0, 0, 4]))

    def test_many_conditions_true(self):
        transition = Transition(1, 2)
        transition.add_condition(Condition(1, operator.lt, 5))
        transition.add_condition(Condition(2, operator.eq, 4))
        transition.add_condition(Condition(2, operator.lt, 5))
        transition.add_condition(Condition(0, operator.lt, 4))
        transition.add_condition(Condition(1, operator.eq, 0))
        transition.add_condition(Condition(0, operator.lt, 4))
        self.assertTrue(transition.check_transition([0, 0, 4]))


class TestCondition(unittest.TestCase):

    def test_valid_condition(self):
        condition = Condition(3, operator.lt, 1)
        self.assertTrue(condition.compare([10, 10, 10, 0.5]))

    def test_invalid_condition(self):
        condition = Condition(2, operator.gt, 1)
        self.assertFalse(condition.compare([10, 10, 0, 0.5]))

    def test_invalid_op(self):
        self.assertRaises(ValueError, Condition, 2, ">=", 1)

    def test_sensor_out_of_range(self):
        condition = Condition(5, operator.gt, 1)
        self.assertRaises(IndexError, condition.compare, [10, 10, 0, 0.5])


if __name__ == '__main__':
    unittest.main()
