from __future__ import print_function

import neat


# import os


# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

class NotAlmostEqualException(Exception):
    pass


def assert_almost_equal(a, b):
    if abs(a - b) > 1e-6:
        max_abs = max(abs(a), abs(b))
        abs_rel_err = abs(a - b) / max_abs
        if abs_rel_err > 1e-6:
            raise NotAlmostEqualException("{0:.6f} !~= {1:.6f}".format(a, b))


def test_softmax():
    """Test the neat.math_utils.softmax function."""
    # Test data - below is from Wikipedia Softmax_function page.
    test_data = [([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [0.02364054302159139, 0.06426165851049616,
                                                        0.17468129859572226, 0.47483299974438037,
                                                        0.02364054302159139, 0.06426165851049616,
                                                        0.17468129859572226])]

    for test in test_data:
        results_list = list(neat.math_util.softmax(test[0]))
        for a, b in zip(test[1], results_list):
            assert_almost_equal(a, b)

    # softmax_result = list(neat.math_util.softmax([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]))
    # print("Softmax for [1, 2, 3, 4, 1, 2, 3] is {!r}".format(softmax_result))


if __name__ == '__main__':
    test_softmax()
