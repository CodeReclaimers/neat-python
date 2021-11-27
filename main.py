from ctrnn_cpg import unit_tests
from ctrnn_cpg import evolve_cpg

#unit_tests.test_min_pop()
#unit_tests.test_eval_genome()
#unit_tests.test_frequency_error()


if __name__ == '__main__':

    for i in range(1):
        evolve_cpg.run(i)
