import gzip
import random
import time
import cPickle

from neat.genome import Genome, FFGenome
from neat.genes import NodeGene, ConnectionGene
from neat.species import Species
from neat.math_util import mean, stdev
from neat.diversity import AgedFitnessSharing


class Population(object):
    """ Manages all the species  """

    def __init__(self, config, checkpoint_file=None, initial_population=None,
                 node_gene_type=NodeGene, conn_gene_type=ConnectionGene,
                 diversity_type=AgedFitnessSharing):
        self.config = config
        self.population = None
        self.node_gene_type = node_gene_type
        self.conn_gene_type = conn_gene_type
        self.diversity = diversity_type(self.config)

        if checkpoint_file:
            # Start from a saved checkpoint.
            self.__resume_checkpoint(checkpoint_file)
        else:
            # currently living species
            self.__species = []
            # species history
            self.species_log = []

            # List of statistics for all generations.
            self.avg_fitness_scores = []
            self.most_fit_genomes = []

            if initial_population is None:
                self.__create_population()
            else:
                self.population = initial_population
            self.generation = -1

    def __resume_checkpoint(self, checkpoint):
        '''
        Resumes the simulation from a previous saved point. This is done by swapping out our existing
        __dict__ with the loaded population's.
        '''
        # TODO: Wouldn't it just be better to create a class method to load and return the stored Population
        # object as-is?  I don't know if there are hidden side effects to directly replacing __dict__.
        with gzip.open(checkpoint) as f:
            print 'Resuming from a previous point: %s' % checkpoint
            # when unpickling __init__ is not called again
            previous_pop = cPickle.load(f)
            self.__dict__ = previous_pop.__dict__

            print 'Loading random state'
            random.setstate(cPickle.load(f))

    def __create_checkpoint(self, report):
        """ Saves the current simulation state. """
        if report:
            print 'Creating checkpoint file at generation: %d' % self.generation

        with gzip.open('checkpoint_' + str(self.generation), 'w', compresslevel=5) as f:
            # Write the entire population state.
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
            # Remember the current random number state.
            cPickle.dump(random.getstate(), f, protocol=2)

    def __create_population(self):
        if self.config.feedforward:
            genotypes = FFGenome
        else:
            genotypes = Genome

        self.population = []
        if self.config.fully_connected:
            for i in xrange(self.config.pop_size):
                g = genotypes.create_fully_connected(self.config, self.node_gene_type, self.conn_gene_type)
                self.population.append(g)
        else:
            for i in xrange(self.config.pop_size):
                g = genotypes.create_minimally_connected(self.config, self.node_gene_type, self.conn_gene_type)
                self.population.append(g)

        if self.config.hidden_nodes > 0:
            for g in self.population:
                g.add_hidden_nodes(self.config.hidden_nodes)

    def __repr__(self):
        s = "Population size: %d" % self.config.pop_size
        s += "\nTotal species: %d" % len(self.__species)
        return s

    def __speciate(self, report):
        """ Group chromosomes into species by similarity """
        # Speciate the population
        for individual in self.population:
            found = False
            for s in self.__species:
                if individual.distance(s.representative) < self.config.compatibility_threshold:
                    s.add(individual)
                    found = True
                    break

            if not found:  # create a new species for this lone chromosome
                self.__species.append(Species(individual))

        # python technical note:
        # we need a "working copy" list when removing elements while looping
        # otherwise we might end up having sync issues
        for s in self.__species[:]:
            # this happens when no chromosomes are compatible with the species
            if len(s.members) == 0:
                if report:
                    print "Removing species %d for being empty" % s.ID
                # remove empty species
                self.__species.remove(s)

        self.__set_compatibility_threshold(report)

    def __set_compatibility_threshold(self, report):
        """ Controls compatibility threshold """
        t = self.config.compatibility_threshold
        dt = self.config.compatibility_change
        if len(self.__species) > self.config.species_size:
            t += dt
        elif len(self.__species) < self.config.species_size:
            t = max(0.0, t - dt)

        if self.config.compatibility_threshold != t:
            if report:
                print("Adjusted compatibility threshold to %f" % t)
            self.config.compatibility_threshold = t

    def __log_species(self):
        """ Logging species data for visualizing speciation """
        higher = max([s.ID for s in self.__species])
        temp = []
        for i in xrange(1, higher + 1):
            found_species = False
            for s in self.__species:
                if i == s.ID:
                    temp.append(len(s.members))
                    found_species = True
                    break
            if not found_species:
                temp.append(0)
        self.species_log.append(temp)

    def epoch(self, fitness_function, n, report=True, save_best=False, checkpoint_interval=10,
              checkpoint_generation=None):
        """ Runs NEAT's genetic algorithm for n epochs.

            Keyword arguments:
            report -- show stats at each epoch (default True)
            save_best -- save the best chromosome from each epoch (default False)
            checkpoint_interval -- time in minutes between saving checkpoints (default 10 minutes)
            checkpoint_generation -- time in generations between saving checkpoints
                (default None -- option disabled)
        """
        t0 = time.time()  # for saving checkpoints

        for g in xrange(n):
            self.generation += 1

            if report:
                print '\n ****** Running generation %d ****** \n' % self.generation

            # Evaluate individuals
            fitness_function(self.population)
            # Speciates the population
            self.__speciate(report)

            # Current generation's best chromosome
            self.most_fit_genomes.append(max(self.population))
            # Current population's average fitness
            self.avg_fitness_scores.append(mean([c.fitness for c in self.population]))

            # Print some statistics
            best = self.most_fit_genomes[-1]

            # saves the best chromo from the current generation
            if save_best:
                f = open('best_chromo_' + str(self.generation), 'w')
                cPickle.dump(best, f)
                f.close()

            # Stops the simulation
            if best.fitness > self.config.max_fitness_threshold:
                if report:
                    print '\nBest individual in epoch %s meets fitness threshold - complexity: %s' % (
                        self.generation, best.size())
                break

            # Remove stagnated species and its members (except if it has the best chromosome)
            for s in self.__species[:]:
                if s.no_improvement_age > self.config.max_stagnation:
                    if report:
                        print "\n   Species %2d (with %2d individuals) is stagnated: removing it" \
                              % (s.ID, len(s.members))
                    # removing species
                    self.__species.remove(s)
                    # removing all the species' members
                    # TODO: can be optimized!
                    for c in self.population[:]:
                        if c.species_id == s.ID:
                            self.population.remove(c)

            # Compute spawn levels for each remaining species
            self.diversity.compute_spawn_amount(self.__species)

            # Removing species with spawn amount = 0
            for s in self.__species[:]:
                # This rarely happens
                if s.spawn_amount == 0:
                    if report:
                        print '   Species %2d age %2s removed: produced no offspring' % (s.ID, s.age)
                    for c in self.population[:]:
                        if c.species_id == s.ID:
                            self.population.remove(c)
                            # self.remove(c)
                    self.__species.remove(s)

            # Logging speciation stats
            self.__log_species()

            if report:
                std_dev = stdev([c.fitness for c in self.population])
                print 'Population\'s average fitness: %3.5f stdev: %3.5f' % (self.avg_fitness_scores[-1], std_dev)
                print 'Best fitness: %2.12s - size: %s - species %s - id %s' \
                      % (best.fitness, best.size(), best.species_id, best.ID)
                print 'Species length: %d totaling %d individuals' \
                      % (len(self.__species), sum([len(s.members) for s in self.__species]))
                print 'Species ID       : %s' % [s.ID for s in self.__species]
                print 'Each species size: %s' % [len(s.members) for s in self.__species]
                print 'Amount to spawn  : %s' % [s.spawn_amount for s in self.__species]
                print 'Species age      : %s' % [s.age for s in self.__species]
                print 'Species no improv: %s' % [s.no_improvement_age for s in self.__species]

            # -------------------------- Producing new offspring -------------------------- #
            new_population = []  # next generation's population

            # Spawning new population
            for s in self.__species:
                new_population.extend(s.reproduce(self.config))

            # Controls under or overflow  #
            fill = self.config.pop_size - len(new_population)
            if fill < 0:  # overflow
                if report:
                    print '   Removing %d excess individual(s) from the new population' % -fill
                # TODO: This is dangerous! I can't remove a species' representative!
                new_population = new_population[:fill]  # Removing the last added members

            if fill > 0:  # underflow
                if report:
                    print '   Producing %d more individual(s) to fill up the new population' % fill

                # TODO: what about producing new individuals instead of reproducing?
                # increasing diversity from time to time might help
                while fill > 0:
                    # Selects a random chromosome from population
                    parent1 = random.choice(self.population)
                    # Search for a mate within the same species
                    found = False
                    for c in self.population:
                        # what if c is parent1 itself?
                        if c.species_id == parent1.species_id:
                            child = parent1.crossover(c)
                            new_population.append(child.mutate())
                            found = True
                            break
                    if not found:
                        # If no mate was found, just mutate it
                        new_population.append(parent1.mutate())
                    # new_population.append(chromosome.FFGenome.create_fully_connected())
                    fill -= 1

            assert self.config.pop_size == len(new_population), 'Different population sizes!'
            # Updates current population
            self.population = new_population[:]

            if checkpoint_interval is not None and time.time() > t0 + 60 * checkpoint_interval:
                self.__create_checkpoint(report)
                t0 = time.time()  # updates the counter
            elif checkpoint_generation is not None and self.generation % checkpoint_generation == 0:
                self.__create_checkpoint(report)
