from neat import config, nn, population


class SequentialStatic(object):
    """
    SequentialStatic is a convenience wrapper to support this common use case:
        "I have some very neat tabular numerical data with input values and
        expected output values, and I'd like a feed-forward network that
        predicts output values based only on a single set of input values."

    This implementation assumes that all hardware cores may be used when possible.
    """

    def __init__(self, input_records, output_records, cfg=None):
        self.input_records = input_records
        self.output_records = output_records
        self.total_evaluations = 0
        self.best_score = 1e38
        self.best_seen = None

        if cfg is None:
            cfg = config.Config()

        self.config = cfg
        num_inputs = len(input_records[0])
        num_outputs = len(output_records[0])
        self.config.genome_config.set_input_output_sizes(num_inputs, num_outputs)

        self.pop = population.Population(self.config)

        # TODO: Default to parallel processing using auto-detected # hardware cores.
        # TODO: Review common customizations and make this wrapper support them in a friendly way when possible.

    def eval_fitness(self, genomes, config):
        self.total_evaluations += len(genomes)

        for g_id, g in genomes:
            net = nn.create_feed_forward_phenotype(g, config)

            sum_square_error = 0.0
            for inputs, expected in zip(self.input_records, self.output_records):
                # Serial activation propagates the inputs through the entire network.
                output = net.serial_activate(inputs)
                for o, e in zip(output, expected):
                    sum_square_error += (o - e) ** 2

            if sum_square_error < self.best_score:
                self.best_score = sum_square_error
                self.best_seen = g

            # When the output matches expected for all inputs, fitness will reach
            # its maximum value of 0.0.
            g.fitness = -sum_square_error

    def evolve(self, num_generations):
        self.pop.run(self.eval_fitness, num_generations)
        return self.pop.statistics.best_genome()

    def evaluate(self, genome):
        net = nn.create_feed_forward_phenotype(genome, self.config)
        for inputs, expected in zip(self.input_records, self.output_records):
            output = net.serial_activate(inputs)
            yield inputs, expected, output
