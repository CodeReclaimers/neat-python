#!/usr/bin/env python
#**************************************************#
# A simple script to help in executing the         #
# same experiment for a number of times.           #
# Please kindly note: THIS IS FAR FROM FINISHED!   #
#**************************************************#
import math, sys, random, pickle
import re, os

p = re.compile('\d*\d')

total_gens = []
total_nodes = []
total_conns = []
total_evals = []

def average(values):
    ''' Returns the population average '''
    sum = 0.0
    for i in values:
        sum += i
    return sum/len(values)

def stdev(values):
    ''' Returns the population standard deviation '''
    # first compute the average
    u = average(values)
    error = 0.0
    # now compute the distance from average
    for x in values:
        error += (u - x)**2
    return math.sqrt(error/len(values))

def report():
    print "\nNumber of runs: %s\n" %sys.argv[2]
    print "\t Gen. \t Nodes \t Conn. \t Evals.\n"
    print "average  %3.2f \t %2.2f \t %2.2f \t %2.2f" \
            %(average(total_gens), average(total_nodes), average(total_conns), average(total_evals))
    print "stdev    %3.2f \t %2.2f \t %2.2f \t %2.2f" \
            %(stdev(total_gens), stdev(total_nodes), stdev(total_conns), stdev(total_evals))

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "\nUsage: run.py experiment.py number_of_runs\n"
        sys.exit(0)

    print "\nExecuting %s for %s times\n" %(sys.argv[1], sys.argv[2])
    print "    =============================================="
    print "\t N. \tGen. \t Nodes \t Conn. \t Evals."

    for i in xrange(int(sys.argv[2])):
        output = os.popen('python '+sys.argv[1]).read()
        try:
            gens, nodes, conns, evals = p.findall(output)
        except: # if anything goes wrong
            if len(output) == 0:
                print "Maximum number of generations reached - got stuck"

        total_gens.append(float(gens))
        total_nodes.append(float(nodes))
        total_conns.append(float(conns))
        total_evals.append(float(evals))
        sys.stdout.flush()
        print "\t %d \t %s \t %s \t %s \t %s" % (i+1, gens, nodes, conns, evals)

    print "    =============================================="
    report()



