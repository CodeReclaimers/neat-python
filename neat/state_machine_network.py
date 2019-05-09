import numpy as np
import operator
from random import choice

from neat.activations import identity_activation
from neat.aggregations import sum_aggregation


class StateMachineNetwork(object):
    """ This class represents a working state machine which can actually run on robot or in simulation. """
    def __init__(self, states, transitions):
        """ Parameters:
            states : dict() where states are sorted based on state_id
            transitions : dict() where transitions are sorted based on begin_id.
        """
        self.states = dict()
        self.transitions = dict()

        # Add the states in the dictionary for easy look-up.
        for state in states:

            if state.id in self.states:
                raise ValueError("State included twice")

            self.states[state.id] = state

            # Add the possibility of transitions from this state.
            if state.id not in self.transitions:
                self.transitions[state.id] = []

        # Add all the transitions indexed by the begin state, for easy lookup.
        for transition in transitions:

            if transition.begin_state_id not in list(self.transitions.keys()):
                raise ValueError('Begin state of transition not in state machine')

            self.transitions[transition.begin_state_id].append(transition)

    def activate(self, current_state_id, inputs):
        """
         :parameter current_state_id : current node of the state machine from where calculations should be done.
         :parameter inputs : Inputs sets, should match the number of inputs to the neural networks.
         :return next_state, output : next state the controller goes to after this execution.
         Output from the current neural network of the controller.
         """

        # First check whether a state transition needs to be made, based on the new data.
        possible_transitions = []
        for transition in self.transitions[current_state_id]:
            if transition.check_transition(inputs):
                possible_transitions.append(transition)

        next_state = current_state_id
        if len(possible_transitions) > 0:
            selected_transition = choice(possible_transitions)
            next_state = selected_transition.end_state_id

        # Evaluate the neural network of the current state.
        current_state = self.states[next_state]
        output = current_state.activate(inputs)

        return next_state, output

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a state machine of neural networks). """

        network_states = []
        for _, state in genome.states.items():
            aggregation_function = config.aggregation_function_defs.get(state.aggregation)
            activation_function = config.activation_defs.get(state.activation)

            network_state = State(state.key, aggregation_function, activation_function)
            network_state.set_biases(state.biases)
            network_state.set_weights(state.weights)
            network_states.append(network_state)

        network_transitions = []
        for _, transition in genome.transitions.items():
            transition_state = Transition(transition.key[0], transition.key[1])
            for condition in transition.conditions:
                transition_state.add_condition(Condition(condition[0], condition[1], condition[2]))

            network_transitions.append(transition_state)

        return StateMachineNetwork(network_states, network_transitions)


class State(object):
    """ This class represents a state in the state machine. """

    def __init__(self, identifier, aggregation_func=sum_aggregation, activation_func=identity_activation):
        """ Default the weights are summed without any function being applied to them."""
        self.id = identifier
        self.biases = None
        self.weights = None
        self.agg_func = aggregation_func
        self.act_func = activation_func
        self.num_inputs = 0
        self.num_outputs = 0

    def set_biases(self, biases):
        """ Enter a list containing the biases of the network (same length as number of outputs) """
        self.biases = np.array(biases)
        self.num_outputs = len(biases)

    def set_weights(self, weights):
        # length rows are #inputs, length columns are #outputs.
        self.weights = np.array(weights)

        self.num_inputs = len(weights[0])

    def activate(self, inputs):

        # Check that the inputs and the weightmatrix are of the same length.
        assert len(inputs) == self.num_inputs

        # Perform neural network operations.
        np_inputs = np.array(inputs)
        combined_weights = np.multiply(np_inputs, self.weights)

        # Note that this sum fails in case of a single row of weight
        aggregate = [self.act_func(self.agg_func(weight_row)) for weight_row in combined_weights]

        assert len(aggregate) == self.num_outputs
        return (aggregate + self.biases).tolist()


class Transition(object):
    """" This class represents a transition in the state machine"""

    def __init__(self, begin_state_id, end_state_id):
        self.begin_state_id = begin_state_id
        self.end_state_id = end_state_id
        self.conditions = []

    def add_condition(self, condition):

        assert isinstance(condition, Condition)
        self.conditions.append(condition)

    def check_transition(self, inputs):
        """ This function checks whether all conditions of the transition holds."""
        evaluations = map(lambda x: x.compare(inputs), self.conditions)
        return all(evaluations)


class Condition(object):
    """ This class represents a condition which can be checked for making a transition. """

    ops = (
        operator.eq,
        operator.gt,
        operator.lt,
    )

    def __init__(self, sensor_id, op, comparison_value):
        if op not in Condition.ops:
            raise ValueError('Invalid operator given to condition')

        self.sensor_id = sensor_id
        self.operator = op
        self.comparison_value = comparison_value

    def compare(self, inputs):
        """ Compares the value of the indicated sensor with the reference value."""
        value = inputs[self.sensor_id]
        return self.operator(value, self.comparison_value)

    @staticmethod
    def random_operator():
        return choice(list(Condition.ops))

    @staticmethod
    def op_to_int(op):
        return Condition.ops.index(op)

    @staticmethod
    def int_to_op(index):
        return Condition.ops[index]
