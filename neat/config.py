# sets the configuration parameters for NEAT
from ConfigParser import ConfigParser


def load(file):
    try:
        config_file = open(file, 'r')
    except IOError:
        print 'Error: file %s not found!' % file
        raise
    else:
        parameters = ConfigParser()
        parameters.readfp(config_file)

        # set class attributes
        # phenotype
        Config.input_nodes          =       int(parameters.get('phenotype','input_nodes'))
        Config.output_nodes         =       int(parameters.get('phenotype','output_nodes'))
        Config.hidden_nodes         =       int(parameters.get('phenotype','hidden_nodes'))
        #print 'fully_connected:',parameters.get('phenotype', 'fully_connected')
        Config.fully_connected      =  bool(int(parameters.get('phenotype','fully_connected')))
        Config.max_weight           =     float(parameters.get('phenotype','max_weight'))
        Config.min_weight           =     float(parameters.get('phenotype','min_weight'))
        Config.feedforward          =  bool(int(parameters.get('phenotype','feedforward')))
        Config.nn_activation        =           parameters.get('phenotype','nn_activation')  # exp or tanh
        Config.weight_stdev         =     float(parameters.get('phenotype','weight_stdev'))

        # GA
        Config.pop_size                 =   int(parameters.get('genetic','pop_size'))
        Config.max_fitness_threshold    = float(parameters.get('genetic','max_fitness_threshold'))
        Config.prob_addconn             = float(parameters.get('genetic','prob_addconn'))
        Config.prob_addnode             = float(parameters.get('genetic','prob_addnode'))
        Config.prob_mutatebias          = float(parameters.get('genetic','prob_mutatebias'))
        Config.bias_mutation_power      = float(parameters.get('genetic','bias_mutation_power'))
        Config.prob_mutate_weight       = float(parameters.get('genetic','prob_mutate_weight'))
        Config.weight_mutation_power    = float(parameters.get('genetic','weight_mutation_power'))
        Config.prob_togglelink          = float(parameters.get('genetic','prob_togglelink'))
        Config.elitism                  = float(parameters.get('genetic','elitism'))

        # genotype compatibility
        Config.compatibility_threshold  = float(parameters.get('genotype compatibility','compatibility_threshold'))
        Config.compatibility_change     = float(parameters.get('genotype compatibility','compatibility_change'))
        Config.excess_coefficient        = float(parameters.get('genotype compatibility','excess_coefficient'))
        Config.disjoint_coefficient      = float(parameters.get('genotype compatibility','disjoint_coefficient'))
        Config.weight_coefficient        = float(parameters.get('genotype compatibility','weight_coefficient'))

        # species
        Config.species_size         =   int(parameters.get('species','species_size'))
        Config.survival_threshold   = float(parameters.get('species','survival_threshold'))
        Config.old_threshold        =   int(parameters.get('species','old_threshold'))
        Config.youth_threshold      =   int(parameters.get('species','youth_threshold'))
        Config.old_penalty          = float(parameters.get('species','old_penalty'))    # always in (0,1)
        Config.youth_boost          = float(parameters.get('species','youth_boost'))   # always in (1,2)
        Config.max_stagnation       =   int(parameters.get('species','max_stagnation'))

class Config:

    # phenotype config
    input_nodes         = None
    output_nodes        = None
    hidden_nodes        = None
    fully_connected     = None
    max_weight          = None
    min_weight          = None
    feedforward         = None
    nn_activation       = None
    weight_stdev        = None

    # GA config
    pop_size                = None
    max_fitness_threshold   = None
    prob_addconn            = None
    prob_addnode            = None
    prob_mutatebias         = None
    bias_mutation_power     = None
    prob_mutate_weight      = None # dynamic mutation rate (future release)
    weight_mutation_power   = None
    prob_togglelink         = None
    elitism                 = None

    #prob_crossover = 0.7  # not implemented (always apply crossover)
    #prob_weightreplaced = 0.0 # not implemented

    # genotype compatibility
    compatibility_threshold = None
    compatibility_change    = None
    excess_coefficient       = None
    disjoint_coefficient     = None
    weight_coefficient       = None

    # species
    species_size        = None
    survival_threshold  = None # only the best 20% for each species is allowed to mate
    old_threshold       = None
    youth_threshold     = None
    old_penalty         = None    # always in (0,1)
    youth_boost         = None    # always in (1,2)
    max_stagnation      = None

    # for a future release
    #ele_event_time = 1000
    #ele_events = False
