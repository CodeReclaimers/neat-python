from examples.experiment_template import SingleExperiment


class DangerZoneExperiment(SingleExperiment):
    """ This class adds the custom functions that need to be applied in a danger zone experiment.
        In this case this is the plotting of the trajectory.
    """

    def output_winner(self):
        super().output_winner()

        self.exp_runner.render = True

        self.exp_runner.run(self.winner, self.learning_config)
        self.exp_runner.env.produce_trajectory('trajectory' + str(self.exp_name) + '.png')

        self.exp_runner.render = False
