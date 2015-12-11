#!/usr/bin/env python
# **************************************************#
# A simple script to help in executing the         #
# same experiment for a number of times.           #
# **************************************************#
import sys
import re
import os

from neat.math_util import mean, stdev

p = re.compile('\d*\d')

total_gens = []
total_nodes = []
total_conns = []
total_evals = []
total_score = []


def report():
    print "\nNumber of runs: {0!s}\n".format(sys.argv[2])
    print "\t Gen. \t Nodes \t Conn. \t Evals. \t Score \n"
    print "average  {0:3.2f} \t {1:2.2f} \t {2:2.2f} \t {3:2.2f} \t {4:2.2f}".format(mean(total_gens), mean(total_nodes), mean(total_conns), mean(total_evals), mean(total_score))
    print "stdev    {0:3.2f} \t {1:2.2f} \t {2:2.2f} \t {3:2.2f} \t {4:2.2f}".format(stdev(total_gens), stdev(total_nodes), stdev(total_conns), stdev(total_evals), stdev(total_score))


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "\nUsage: run.py experiment.py number_of_runs\n"
        sys.exit(0)

    print "\nExecuting {0!s} for {1!s} times\n".format(sys.argv[1], sys.argv[2])
    print "    =========================================================="
    print "\t N. \tGen. \t Nodes \t Conn. \t Evals.    Score"

    for i in xrange(int(sys.argv[2])):
        output = os.popen('python ' + sys.argv[1]).read()
        try:
            gens, nodes, conns, evals, score = p.findall(output)
        except:  # if anything goes wrong
            print output
            if len(output) == 0:
                print "Maximum number of generations reached - got stuck"

        total_gens.append(float(gens))
        total_nodes.append(float(nodes))
        total_conns.append(float(conns))
        total_evals.append(float(evals))
        total_score.append(float(score))
        sys.stdout.flush()
        print "\t {0:d} \t {1!s} \t {2!s} \t {3!s} \t {4!s} \t {5!s}".format(i + 1, gens, nodes, conns, evals, score)

    print "    =========================================================="
    report()
