# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# This is a work in progress, and currently takes ~100 generations to
# find a network that can land with a score >= 200 at least a couple of
# times.  It has yet to solve the environment, which could have something
# to do to me being totally clueless in regard to reinforcement learning. :)

from __future__ import print_function

import gym
import gym.wrappers

import matplotlib.pyplot as plt

import neat
import numpy as np
import os
import pickle
import random

import visualize

env = gym.make('LunarLander-v2')

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))

# Limit episodes to 400 time steps to cut down on training time.
# 400 steps is more than enough time to land with a winning score.
print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 400
print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))

env = gym.wrappers.Monitor(env, 'results', force=True)

discounted_reward = 0.9
min_reward = -200
max_reward = 200

score_range = []

def eval_fitness_shared(genomes, config):
    nets = []
    for gid, g in genomes:
        nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))
        g.fitness = []

    episodes = []
    scores = []
    for genome, net in nets:
        observation = env.reset()
        episode_data = []
        j = 0
        total_score = 0.0
        while 1:
            if net is not None:
                output = net.activate(observation)
                action = np.argmax(output)
            else:
                action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            total_score += reward
            episode_data.append((j, observation, action, reward))

            if done:
                break

            j += 1

        episodes.append(episode_data)
        scores.append(total_score)
        genome.fitness = total_score

    if scores:
        score_range.append((min(scores), np.mean(scores), max(scores)))

    # Compute discounted rewards.
    discounted_rewards = []
    for episode in episodes:
        rewards = np.array([reward for j, observation, action, reward in episode])
        N = len(episode)
        D = np.sum((np.eye(N, k=i) * discounted_reward ** i for i in range(N)))
        discounted_rewards.append(np.dot(D, rewards))

    print(min(map(np.min, discounted_rewards)), max(map(np.max, discounted_rewards)))

    # Normalize rewards
    for i in range(len(discounted_rewards)):
        discounted_rewards[i] = 2 * (discounted_rewards[i] - min_reward) / (max_reward - min_reward) - 1.0

    print(min(map(np.min, discounted_rewards)), max(map(np.max, discounted_rewards)))

    episode_filter = [random.randint(0, len(episodes)-1) for _ in range(10)]
    for genome, net in nets:
        reward_error = []
        for i in episode_filter:
            episode = episodes[i]
            discount_reward = discounted_rewards[i]
            for (j, observation, action, reward), dr in zip(episode, discount_reward):
                #test_set.append((observation, action, reward, dr))
                output = net.activate(observation)
                reward_error.append((output[action] - dr)**2)

        print(genome.fitness, np.mean(reward_error))
        genome.fitness -= 100 * np.mean(reward_error)


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
    pop.add_reporter(neat.StdOutReporter())
    # Checkpoint every 10 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(10, 900))

    # Run until the winner from a generation is able to solve the environment.
    while 1:
        winner = pop.run(eval_fitness_shared, 1)

        visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

        if score_range:
            S = np.array(score_range).T
            plt.plot(S[0], 'r-')
            plt.plot(S[1], 'b-')
            plt.plot(S[2], 'g-')
            plt.grid()
            plt.savefig("score-ranges.svg")
            plt.close()

        mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
        print("Average mean fitness over last 5 generations: {0}".format(mfs))

        mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
        print("Average min fitness over last 5 generations: {0}".format(mfs))

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        for k in range(100):
            observation = env.reset()
            score = 0
            while 1:
                output = winner_net.activate(observation)
                observation, reward, done, info = env.step(np.argmax(output))
                score += reward
                env.render()
                if done:
                    break
            print(k, score)
            if score < 200:
                break
        else:
            print("Solved.")
            break

    winner = stats.best_genome()
    print(winner)

    # Save the winner.
    with open('winner.pickle', 'wb') as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, ylog=False, view=True, filename="fitness.svg")
    visualize.plot_species(stats, view=True, filename="speciation.svg")

    visualize.draw_net(config, winner, True)

    visualize.draw_net(config, winner, view=True, filename="winner-net.gv")
    visualize.draw_net(config, winner, view=True, filename="winner-net-enabled.gv",
                       show_disabled=False)
    visualize.draw_net(config, winner, view=True, filename="winner-net-enabled-pruned.gv",
                       show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()