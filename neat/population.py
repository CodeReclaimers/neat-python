import gzip
import random
import math
import time
import cPickle as pickle
from config import Config
import species
import chromosome


class Population(object):
    """ Manages all the species  """

    def __init__(self, checkpoint_file=None, initial_population=None):
        self.population = None

        if checkpoint_file:
            # start from a previous point: creates an 'empty'
            # population and point its __dict__ to the previous one
            self.__resume_checkpoint(checkpoint_file)
        else:
            # total population size
            self.__popsize = Config.pop_size
            # currently living species
            self.__species = []
            # species history
            self.__species_log = []

            # Statistics
            self.__avg_fitness = []
            self.__best_fitness = []

            if initial_population is None:
                self.__create_population()
            else:
                self.population = initial_population
            self.generation = -1

    def stats(self):
        return (self.__best_fitness, self.__avg_fitness)

    def species_log(self):
        return self.__species_log

    def __resume_checkpoint(self, checkpoint):
        """ Resumes the simulation from a previous saved point. """
        try:
            # file = open(checkpoint)
            f = gzip.open(checkpoint)
        except IOError:
            raise
        print 'Resuming from a previous point: %s' % checkpoint
        # when unpickling __init__ is not called again
        previous_pop = pickle.load(f)
        self.__dict__ = previous_pop.__dict__

        print 'Loading random state'
        rstate = pickle.load(f)
        random.setstate(rstate)
        # random.jumpahead(1)
        f.close()

    def __create_checkpoint(self, report):
        """ Saves the current simulation state. """
        # from time import strftime
        # get current time
        # date = strftime("%Y_%m_%d_%Hh%Mm%Ss")
        if report:
            print 'Creating checkpoint file at generation: %d' % self.generation

        # dumps 'self'
        # file = open('checkpoint_'+str(self.generation), 'w')
        f = gzip.open('checkpoint_' + str(self.generation), 'w', compresslevel=5)
        # dumps the population
        pickle.dump(self, f, protocol=2)
        # dumps the current random state
        pickle.dump(random.getstate(), f, protocol=2)
        f.close()

    def __create_population(self):

        if Config.feedforward:
            genotypes = chromosome.FFChromosome
        else:
            genotypes = chromosome.Chromosome

        self.population = []
        for i in xrange(self.__popsize):
            g = genotypes.create_fully_connected() \
                if Config.fully_connected \
                else genotypes.create_minimally_connected()
            if Config.hidden_nodes > 0:
                g.add_hidden_nodes(Config.hidden_nodes)
            self.population.append(g)

    def __repr__(self):
        s = "Population size: %d" % self.__popsize
        s += "\nTotal species: %d" % len(self.__species)
        return s

    def __speciate(self, report):
        """ Group chromosomes into species by similarity """
        # Speciate the population
        for individual in self.population:
            found = False
            for s in self.__species:
                if individual.distance(s.representant) < Config.compatibility_threshold:
                    s.add(individual)
                    found = True
                    break

            if not found:  # create a new species for this lone chromosome
                self.__species.append(species.Species(individual))

        # python technical note:
        # we need a "working copy" list when removing elements while looping
        # otherwise we might end up having sync issues
        for s in self.__species[:]:
            # this happens when no chromosomes are compatible with the species
            if len(s) == 0:
                if report:
                    print "Removing species %d for being empty" % s.id
                # remove empty species
                self.__species.remove(s)

        self.__set_compatibility_threshold()

    def __set_compatibility_threshold(self):
        """ Controls compatibility threshold """
        if len(self.__species) > Config.species_size:
            Config.compatibility_threshold += Config.compatibility_change
        elif len(self.__species) < Config.species_size:
            if Config.compatibility_threshold > Config.compatibility_change:
                Config.compatibility_threshold -= Config.compatibility_change
            else:
                print 'Compatibility threshold cannot be changed (minimum value has been reached)'

    def average_fitness(self):
        """ Returns the average raw fitness of population """
        s = 0.0
        for c in self.population:
            s += c.fitness

        return s / len(self.population)

    def stdeviation(self):
        """ Returns the population standard deviation """
        # first compute the average
        u = self.average_fitness()
        error = 0.0

        try:
            # now compute the distance from average
            for c in self.population:
                error += (u - c.fitness) ** 2
        except OverflowError:
            # TODO: catch OverflowError: (34, 'Numerical result out of range')
            print "Overflow - printing population status"
            print "error = %f \t average = %f" % (error, u)
            print "Population fitness:"
            print [c.fitness for c in self]

        return math.sqrt(error / len(self.population))

    def __compute_spawn_levels(self):
        """ Compute each species' spawn amount (Stanley, p. 40) """

        # 1. Boost if young and penalize if old
        # TODO: does it really increase the overall performance?
        species_stats = []
        for s in self.__species:
            if s.age < Config.youth_threshold:
                species_stats.append(s.average_fitness() * Config.youth_boost)
            elif s.age > Config.old_threshold:
                species_stats.append(s.average_fitness() * Config.old_penalty)
            else:
                species_stats.append(s.average_fitness())

        # 2. Share fitness (only usefull for computing spawn amounts)
        # More info: http://tech.groups.yahoo.com/group/neat/message/2203
        # Sharing the fitness is only meaningful here
        # we don't really have to change each individual's raw fitness
        total_average = 0.0
        for s in species_stats:
            total_average += s

            # 3. Compute spawn
        for i, s in enumerate(self.__species):
            s.spawn_amount = int(round((species_stats[i] * self.__popsize / total_average)))

    def __tournament_selection(self, k=2):
        """ Tournament selection with size k (default k=2).
            Make sure the population has at least k individuals """
        random.shuffle(self.population)

        return max(self.population[:k])

    def __log_species(self):
        """ Logging species data for visualizing speciation """
        higher = max([s.id for s in self.__species])
        temp = []
        for i in xrange(1, higher + 1):
            found_specie = False
            for s in self.__species:
                if i == s.id:
                    temp.append(len(s))
                    found_specie = True
                    break
            if not found_specie:
                temp.append(0)
        self.__species_log.append(temp)

    def __population_diversity(self):
        """ Calculates the diversity of population: total average weights,
            number of connections, nodes """

        num_nodes = 0
        num_conns = 0
        avg_weights = 0.0

        for c in self.population:
            num_nodes += len(c.node_genes)
            num_conns += len(c.conn_genes)
            for cg in c.conn_genes:
                avg_weights += cg.weight

        total = len(self.population)
        return (num_nodes / total, num_conns / total, avg_weights / total)

    def epoch(self, fitness_function, n, report=True, save_best=False, checkpoint_interval=10,
              checkpoint_generation=None):
        """ Runs NEAT's genetic algorithm for n epochs.

            Keyword arguments:
            report -- show stats at each epoch (default True)
            save_best -- save the best chromosome from each epoch (default False)
            checkpoint_interval -- time in minutes between saving checkpoints (default 10 minutes)
            checkpoint_generation -- time in generations between saving checkpoints
                (default 0 -- option disabled)
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
            self.__best_fitness.append(max(self.population))
            # Current population's average fitness
            self.__avg_fitness.append(self.average_fitness())

            # Print some statistics
            best = self.__best_fitness[-1]
            # Which species has the best chromosome?
            for s in self.__species:
                s.hasBest = False
                if best.species_id == s.id:
                    s.hasBest = True

            # saves the best chromo from the current generation
            if save_best:
                f = open('best_chromo_' + str(self.generation), 'w')
                pickle.dump(best, f)
                f.close()

            # Stops the simulation
            if best.fitness > Config.max_fitness_threshold:
                print '\nBest individual in epoch %s meets fitness threshold - complexity: %s' % (
                self.generation, best.size())
                break

            # -----------------------------------------
            # Prints chromosome's parents id:  {dad_id, mom_id} -> child_id
            # for chromosome in self.population:
            #    print '{%3d; %3d} -> %3d' %(chromosome.parent1_id, chromosome.parent2_id, chromosome.id)
            # -----------------------------------------

            # Remove stagnated species and its members (except if it has the best chromosome)
            for s in self.__species[:]:
                if s.no_improvement_age > Config.max_stagnation:
                    if not s.hasBest:
                        if report:
                            print "\n   Species %2d (with %2d individuals) is stagnated: removing it" \
                                  % (s.id, len(s))
                        # removing species
                        self.__species.remove(s)
                        # removing all the species' members
                        # TODO: can be optimized!
                        for c in self.population[:]:
                            if c.species_id == s.id:
                                self.population.remove(c)

            # Remove "super-stagnated" species (even if it has the best chromosome)
            # It is not clear if it really avoids local minima
            for s in self.__species[:]:
                if s.no_improvement_age > 2 * Config.max_stagnation:
                    if report:
                        print "\n   Species %2d (with %2d individuals) is super-stagnated: removing it" \
                              % (s.id, len(s))
                    # removing species
                    self.__species.remove(s)
                    # removing all the species' members
                    # TODO: can be optimized!
                    for c in self.population[:]:
                        if c.species_id == s.id:
                            self.population.remove(c)

            # Compute spawn levels for each remaining species
            self.__compute_spawn_levels()

            # Removing species with spawn amount = 0
            for s in self.__species[:]:
                # This rarely happens
                if s.spawn_amount == 0:
                    if report:
                        print '   Species %2d age %2s removed: produced no offspring' % (s.id, s.age)
                    for c in self.population[:]:
                        if c.species_id == s.id:
                            self.population.remove(c)
                            # self.remove(c)
                    self.__species.remove(s)

            # Logging speciation stats
            self.__log_species()

            if report:
                # print 'Poluation size: %d \t Divirsity: %s' %(len(self.population), self.__population_diversity())
                print 'Population\'s average fitness: %3.5f stdev: %3.5f' % (self.__avg_fitness[-1], self.stdeviation())
                print 'Best fitness: %2.12s - size: %s - species %s - id %s' \
                      % (best.fitness, best.size(), best.species_id, best.id)

                # print some "debugging" information
                print 'Species length: %d totalizing %d individuals' \
                      % (len(self.__species), sum([len(s) for s in self.__species]))
                print 'Species ID       : %s' % [s.id for s in self.__species]
                print 'Each species size: %s' % [len(s) for s in self.__species]
                print 'Amount to spawn  : %s' % [s.spawn_amount for s in self.__species]
                print 'Species age      : %s' % [s.age for s in self.__species]
                print 'Species no improv: %s' % [s.no_improvement_age for s in
                                                 self.__species]  # species no improvement age

                # for s in self.__species:
                #    print s

            # -------------------------- Producing new offspring -------------------------- #
            new_population = []  # next generation's population

            # Spawning new population
            for s in self.__species:
                new_population.extend(s.reproduce())

            # ----------------------------#
            # Controls under or overflow  #
            # ----------------------------#
            fill = self.__popsize - len(new_population)
            if fill < 0:  # overflow
                if report: print '   Removing %d excess individual(s) from the new population' % -fill
                # TODO: This is dangerous! I can't remove a species' representant!
                new_population = new_population[:fill]  # Removing the last added members

            if fill > 0:  # underflow
                if report: print '   Producing %d more individual(s) to fill up the new population' % fill

                # TODO:
                # what about producing new individuals instead of reproducing?
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
                    # new_population.append(chromosome.FFChromosome.create_fully_connected())
                    fill -= 1

            assert self.__popsize == len(new_population), 'Different population sizes!'
            # Updates current population
            self.population = new_population[:]

            if checkpoint_interval is not None and time.time() > t0 + 60 * checkpoint_interval:
                self.__create_checkpoint(report)
                t0 = time.time()  # updates the counter
            elif checkpoint_generation is not None and self.generation % checkpoint_generation == 0:
                self.__create_checkpoint(report)


if __name__ == '__main__':

    # sample fitness function
    def eval_fitness(population):
        for individual in population:
            individual.fitness = 1.0


    # creates the population
    pop = Population()
    # runs the simulation for 250 epochs
    pop.epoch(eval_fitness, 250)
