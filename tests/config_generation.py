from neat.state_machine_genome import StateMachineGenome
from neat.state_machine_genome_partially_fixed import StateMachineGenomeFixed


def init_config():
    params = dict()
    params['num_inputs'] = 3
    params['num_outputs'] = 2
    params['num_initial_states'] = 1
    params['state_add_prob'] = 1
    params['state_delete_prob'] = 1
    params['transition_add_prob'] = 1
    params['transition_delete_prob'] = 1

    params['activation'] = 'sigmoid'
    params['aggregation'] = 'sum'

    params['compatibility_disjoint_coefficient'] = 0.5
    params['compatibility_difference_coefficient'] = 1.0

    params['weight_init_mean'] = 1
    params['weight_init_stdev'] = 1
    params['weight_replace_rate'] = 0.1
    params['weight_mutate_rate'] = 0.5
    params['weight_mutate_power'] = 1
    params['weight_max_value'] = 2
    params['weight_min_value'] = 0

    params['bias_init_mean'] = 1
    params['bias_init_stdev'] = 1
    params['bias_replace_rate'] = 0.1
    params['bias_mutate_rate'] = 0.5
    params['bias_mutate_power'] = 1
    params['bias_max_value'] = 2
    params['bias_min_value'] = 0

    params['conditions_add_condition_prob'] = 1
    params['conditions_remove_condition_prob'] = 1
    params['condition_mutate_input_prob'] = 1
    params['condition_mutate_comp_prob'] = 1

    params['condition_comparator_init_mean'] = 1
    params['condition_comparator_init_stdev'] = 1
    params['condition_comparator_replace_rate'] = 0.1
    params['condition_comparator_mutate_rate'] = 0.5
    params['condition_comparator_mutate_power'] = 1
    params['condition_comparator_max_value'] = 2
    params['condition_comparator_min_value'] = 0

    params['enabled_default'] = 'true'
    params['enabled_mutate_rate'] = 1

    config = StateMachineGenome.parse_config(params)

    return config


def init_fixed_genome_config():

    params = dict()
    params['num_inputs'] = 1
    params['num_outputs'] = 1
    params['num_initial_states'] = 1
    params['state_add_prob'] = 1
    params['state_delete_prob'] = 1
    params['transition_add_prob'] = 1
    params['transition_delete_prob'] = 1

    params['activation'] = 'sigmoid'
    params['aggregation'] = 'sum'

    params['compatibility_disjoint_coefficient'] = 0.5
    params['compatibility_difference_coefficient'] = 1.0

    params['weight_init_mean'] = 1
    params['weight_init_stdev'] = 1
    params['weight_replace_rate'] = 0.1
    params['weight_mutate_rate'] = 0.5
    params['weight_mutate_power'] = 1
    params['weight_max_value'] = 2
    params['weight_min_value'] = 0

    params['bias_init_mean'] = 1
    params['bias_init_stdev'] = 1
    params['bias_replace_rate'] = 0.1
    params['bias_mutate_rate'] = 0.5
    params['bias_mutate_power'] = 1
    params['bias_max_value'] = 2
    params['bias_min_value'] = 0

    params['conditions_add_condition_prob'] = 1
    params['conditions_remove_condition_prob'] = 1
    params['condition_mutate_input_prob'] = 1
    params['condition_mutate_comp_prob'] = 1

    params['condition_comparator_init_mean'] = 1
    params['condition_comparator_init_stdev'] = 1
    params['condition_comparator_replace_rate'] = 0.1
    params['condition_comparator_mutate_rate'] = 0.5
    params['condition_comparator_mutate_power'] = 1
    params['condition_comparator_max_value'] = 2
    params['condition_comparator_min_value'] = 0

    params['enabled_default'] = 'true'
    params['enabled_mutate_rate'] = 1

    params['genome_source'] = 'test_genome.pickle'
    params['fixed_section'] = 'states'

    return StateMachineGenomeFixed.parse_config(params)
