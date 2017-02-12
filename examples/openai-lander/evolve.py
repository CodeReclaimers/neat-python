# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg

from __future__ import print_function

import gym
import gym.wrappers

import matplotlib.pyplot as plt

import multiprocessing
import neat
import numpy as np
import os
import pickle
import random
import time

import visualize

env = gym.make('LunarLander-v2')

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))

env = gym.wrappers.Monitor(env, 'results', force=True)


class LanderGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.discount = None

    def configure_new(self, config):
        super().configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff


def compute_fitness(net, episodes):
    reward_error = []
    for score, observations, acts, rewards in episodes:
        for o, a, r in zip(observations, acts, rewards):
            output = net.activate(o)
            reward_error.append(float((output[a] - r) ** 2))

    return reward_error


class PooledErrorCompute(object):
    def __init__(self):
        self.pool = multiprocessing.Pool()
        self.test_episodes = []

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

    def evaluate_genomes(self, genomes, config):
        t0 = time.time()
        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))
            g.fitness = []

        print("network creation time {0}".format(time.time() - t0))
        t0 = time.time()

        episodes = []
        for genome, net in nets:
            observation = env.reset()
            step = 0
            observations = []
            actions = []
            rewards = []
            while 1:
                step += 1
                if step < 200 and random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    output = net.activate(observation)
                    action = np.argmax(output)

                observation, reward, done, info = env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)

                if done:
                    break

            total_score = sum(rewards)
            self.episode_score.append(total_score)
            self.episode_length.append(step)

            # Compute discounted rewards.
            m = int(round(np.log(0.01) / np.log(genome.discount)))
            discount_function = [genome.discount ** (m - i) for i in range(m + 1)]
            #rewards = np.array([reward for observation, action, reward in episode])
            disc_rewards = np.convolve(rewards, discount_function)[m:]

            # Normalize discounted rewards.
            normed_rewards = 2 * (disc_rewards - self.min_reward) / (self.max_reward - self.min_reward) - 1.0

            episodes.append((total_score, observations, actions, normed_rewards))
            genome.fitness = total_score

        print("simulation run time {0}".format(time.time() - t0))
        t0 = time.time()

        # Randomly choose subset of episodes for evaluation of genome reward estimation.
        #self.test_episodes.extend(random.choice(episodes)[1] for _ in range(20))
        self.test_episodes.extend(episodes)
        #self.test_episodes = [random.choice(self.test_episodes) for _ in range(200)]
        self.test_episodes = self.test_episodes[-1500:]
        eps = [random.choice(self.test_episodes) for _ in range(50)]

        print("Evaluating {0} test episodes".format(len(eps)))

        jobs = []
        for genome, net in nets:
            jobs.append(self.pool.apply_async(compute_fitness, (net, eps)))

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.
        for job, (genome_id, genome) in zip(jobs, genomes):
            reward_error = job.get(timeout=None)
            genome.fitness -= 50 * np.mean(reward_error)

        print("final fitness compute time {0}\n".format(time.time() - t0))


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(LanderGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute()
    while 1:
        try:
            pop.run(ec.evaluate_genomes, 1)

            visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

            plt.plot(ec.episode_score, 'g-', label='score')
            plt.plot(ec.episode_length, 'b-', label='length')
            plt.grid()
            plt.legend(loc='best')
            plt.savefig("scores.svg")
            plt.close()

            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Use the five best genomes seen so far as an ensemble-ish control system.
            best_genomes = stats.best_unique_genomes(5)
            best_networks = []
            for g in best_genomes:
                best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))

            solved = True
            best_scores = []
            for k in range(100):
                observation = env.reset()
                score = 0
                step = 0
                while 1:
                    step += 1
                    # Use the total reward estimates from all five networks to
                    # determine the best action given the current state.
                    total_rewards = np.zeros((4,))
                    for n in best_networks:
                        output = n.activate(observation)
                        total_rewards += output

                    best_action = np.argmax(total_rewards)
                    observation, reward, done, info = env.step(best_action)
                    score += reward
                    env.render()
                    if done:
                        break

                ec.episode_score.append(score)
                ec.episode_length.append(step)

                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                print(k, score, avg_score)
                if avg_score < 200:
                    solved = False
                    break

            if solved:
                print("Solved.")

                # Save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name+'.pickle', 'wb') as f:
                        pickle.dump(g, f)

                    visualize.draw_net(config, g, view=False, filename=name+"-net.gv")
                    visualize.draw_net(config, g, view=False, filename=name+"-net-enabled.gv",
                                       show_disabled=False)
                    visualize.draw_net(config, g, view=False, filename=name+"-net-enabled-pruned.gv",
                                       show_disabled=False, prune_unused=True)

                break
        except KeyboardInterrupt:
            print("User break.")
            break

    env.close()


if __name__ == '__main__':
    run()