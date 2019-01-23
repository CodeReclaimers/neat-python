from neat.state_machine_attributes import SimpleNeuralNetworkAttribute
from neat.state_machine_genome import StateMachineGenome
import timeit


def init_params():
    params = dict()
    params['num_inputs'] = 2
    params['num_outputs'] = 2
    params['state_add_prob'] = 1
    params['state_delete_prob'] = 1
    params['transition_add_prob'] = 1
    params['transition_delete_prob'] = 1

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

    return params


if __name__ == '__main__':

    config_params = init_params()
    config = StateMachineGenome.parse_config(config_params)

    attribute = SimpleNeuralNetworkAttribute('nn')
    nn = attribute.init_value(config)
    print(nn)

    nn1 = attribute.mutate_value(nn, config)
    print(nn1)

    run_time = timeit.timeit('attribute.mutate_value(nn, config)',
                             'from __main__ import config, attribute, nn',
                             number=100000)
    print(run_time)

