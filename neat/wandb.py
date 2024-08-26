# based on reporting.py and statistics.py, make a WandbReporter
import wandb
from neat.reporting import BaseReporter

class WandbReporter(BaseReporter):
    def __init__(self, api_key, project_name, tags=None):
        super().__init__()
        self.api_key = api_key
        self.project_name = project_name
        self.tags = tags


    def start_generation(self, generation):
        wandb.init(project=self.project_name, tags=self.tags)
        wandb.log({"generation": generation})

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        wandb.log({"best_genome": best_genome.fitness})

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass