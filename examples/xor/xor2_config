#--- parameters for the XOR-2 experiment ---#
# Stats for the following parameters averaged (std) over 5000 runs:
#   generations     hidden nodes      synapses
#   26.2 (7.48)       2.9 (1.3)      10.4 (3.37)
# Additional info: error threshold = 0.9
#                  neuron's response = 4.924273

[phenotype]
input_nodes         = 2
output_nodes        = 1
fully_connected     = 1
max_weight          = 30
min_weight          = -30
feedforward         = 1
nn_activation       = exp
hidden_nodes        = 0
weight_stdev        = 0.9

[genetic]
pop_size              = 150
max_fitness_threshold = 0.9

# Human reasoning
prob_addconn          = 0.05
prob_addnode          = 0.03
prob_mutatebias       = 0.20
bias_mutation_power   = 0.50
prob_mutate_weight    = 0.90
weight_mutation_power = 1.50
prob_togglelink       = 0.01
elitism               = 1

# Parameters obtained with a meta-SGA
#prob_addconn          = 0.32
#prob_addnode          = 0.08
#prob_mutatebias       = 0.26
#bias_mutation_power   = 0.21
#prob_mutate_weight    = 0.50
#weight_mutation_power = 0.45
#prob_togglelink       = 0.02

[genotype compatibility]
compatibility_threshold = 3.0
compatibility_change    = 0.0
excess_coeficient       = 1.0
disjoint_coeficient     = 1.0
weight_coeficient       = 0.4

[species]
species_size        = 10
survival_threshold  = 0.2
old_threshold       = 30
youth_threshold     = 10
old_penalty         = 0.2
youth_boost         = 1.2
max_stagnation      = 15


