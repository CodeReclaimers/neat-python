import neat
config_filename_initial = 'test_configuration'
config_filename_save = 'save_configuration'

# Load initial configuration from file
config_initial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, config_filename_initial)

# save configuration to another file
config_initial.save(config_filename_save)

# obtain configuration from saved file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, config_filename_save)
