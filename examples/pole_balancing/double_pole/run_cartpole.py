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
    print "\nNumber of runs: %s\n" % sys.argv[2]
    print "\t Gen. \t Nodes \t Conn. \t Evals. \t Score \n"
    print "average  %3.2f \t %2.2f \t %2.2f \t %2.2f \t %2.2f" \
          % (mean(total_gens), mean(total_nodes), mean(total_conns), mean(total_evals), mean(total_score))
    print "stdev    %3.2f \t %2.2f \t %2.2f \t %2.2f \t %2.2f" \
          % (stdev(total_gens), stdev(total_nodes), stdev(total_conns), stdev(total_evals), stdev(total_score))


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "\nUsage: run.py experiment.py number_of_runs\n"
        sys.exit(0)

    print "\nExecuting %s for %s times\n" % (sys.argv[1], sys.argv[2])
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
        print "\t %d \t %s \t %s \t %s \t %s \t %s" % (i + 1, gens, nodes, conns, evals, score)

    print "    =========================================================="
    report()
