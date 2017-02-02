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

discounted_reward = 0.9
M = int(round(np.log(0.01) / np.log(discounted_reward)))
discount_function = [discounted_reward ** (M - i) for i in range(M + 1)]
print("Discount function window {0}".format(len(discount_function)))

min_reward = -200
max_reward = 200

score_range = []



def compute_fitness(net, discounted_rewards, episodes):
    reward_error = []
    for discount_reward, episode in zip(discounted_rewards, episodes):
        for (observation, action, reward), dr in zip(episode, discount_reward):
            output = net.activate(observation)
            reward_error.append(float((output[action] - dr) ** 2))

    return reward_error


class PooledErrorCompute(object):
    def __init__(self):
        self.pool = multiprocessing.Pool()
        self.test_episodes = []

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
            episode_data = []
            total_score = 0.0
            while 1:
                output = net.activate(observation)
                action = np.argmax(output)
                observation, reward, done, info = env.step(action)
                total_score += reward
                episode_data.append((observation, action, reward))

                if done:
                    break

            episodes.append((total_score, episode_data))
            genome.fitness = total_score

        print("simulation run time {0}".format(time.time() - t0))
        t0 = time.time()

        scores = [s for s, e in episodes]
        score_range.append((min(scores), np.mean(scores), max(scores)))

        # Compute discounted rewards.
        discounted_rewards = []
        for score, episode in episodes:
            rewards = np.array([reward for observation, action, reward in episode])
            discounted = np.convolve(rewards, discount_function)[M:]
            discounted_rewards.append(discounted)

        print(min(map(np.min, discounted_rewards)), max(map(np.max, discounted_rewards)))

        # Normalize rewards
        for i in range(len(discounted_rewards)):
            discounted_rewards[i] = 2 * (discounted_rewards[i] - min_reward) / (max_reward - min_reward) - 1.0

        print(min(map(np.min, discounted_rewards)), max(map(np.max, discounted_rewards)))

        print("discounted & normalized reward compute time {0}".format(time.time() - t0))
        t0 = time.time()

        # Randomly choose subset of episodes for evaluation of genome reward estimation.
        self.test_episodes.extend(random.choice(episodes)[1] for _ in range(20))
        self.test_episodes = [random.choice(self.test_episodes) for _ in range(100)]
        eps = [random.choice(self.test_episodes) for _ in range(20)]

        print("Evaluating {0} test episodes".format(len(eps)))

        jobs = []
        for genome, net in nets:
            jobs.append(self.pool.apply_async(compute_fitness, (net, discounted_rewards, eps)))

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
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
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

            if score_range:
                S = np.array(score_range).T
                plt.plot(S[1], 'b-')
                plt.plot(S[2], 'g-')
                plt.grid()
                plt.savefig("score-ranges.svg")
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
                while 1:
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