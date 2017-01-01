from cart_pole import CartPole, run_simulation

runs_per_net = 5


def evaluate_population(genomes, create_func, force_func):
    for g in genomes:
        net = create_func(g)

        fitness = 0

        for runs in range(runs_per_net):
            sim = CartPole()
            fitness += run_simulation(sim, net, force_func)

        # The genome's fitness is its average performance across all runs.
        g.fitness = fitness / float(runs_per_net)
