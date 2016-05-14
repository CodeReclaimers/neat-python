from neat.activations import sigmoid_activation, tanh_activation, \
    sin_activation, gauss_activation, relu_activation, identity_activation, \
    clamped_activation, inv_activation, log_activation, exp_activation, \
    abs_activation, hat_activation, square_activation, cube_activation


# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

class NotAlmostEqualException(Exception):
    pass


def assert_almost_equal(a, b):
    if abs(a - b) > 1e-6:
        max_abs = max(abs(a), abs(b))
        abs_rel_err = abs(a - b) / max_abs
        if abs_rel_err > 1e-6:
            raise NotAlmostEqualException()


def test_sigmoid():
    assert sigmoid_activation(0.0) == 0.5


def test_tanh():
    assert tanh_activation(0.0) == 0.0


def test_sin():
    assert sin_activation(0.0) == 0.0


def test_gauss():
    assert_almost_equal(gauss_activation(0.0), 0.398942280401)
    assert_almost_equal(gauss_activation(-1.0),
                        gauss_activation(1.0))


def test_relu():
    assert relu_activation(-1.0) == 0.0
    assert relu_activation(0.0) == 0.0
    assert relu_activation(1.0) == 1.0


def test_identity():
    assert identity_activation(-1.0) == -1.0
    assert identity_activation(0.0) == 0.0
    assert identity_activation(1.0) == 1.0


def test_clamped():
    assert clamped_activation(-2.0) == -1.0
    assert clamped_activation(-1.0) == -1.0
    assert clamped_activation(0.0) == 0.0
    assert clamped_activation(1.0) == 1.0
    assert clamped_activation(2.0) == 1.0


def test_inv():
    assert inv_activation(1.0) == 1.0
    assert inv_activation(0.5) == 2.0
    assert inv_activation(2.0) == 0.5
    assert inv_activation(0.0) == 0.0


def test_log():
    assert log_activation(1.0) == 0.0


def test_exp():
    assert exp_activation(0.0) == 1.0


def test_abs():
    assert abs_activation(-1.0) == 1.0
    assert abs_activation(0.0) == 0.0
    assert abs_activation(-1.0) == 1.0


def test_hat():
    assert hat_activation(-1.0) == 0.0
    assert hat_activation(0.0) == 1.0
    assert hat_activation(1.0) == 0.0


def test_square():
    assert square_activation(-1.0) == 1.0
    assert square_activation(-0.5) == 0.25
    assert square_activation(0.0) == 0.0
    assert square_activation(0.5) == 0.25
    assert square_activation(1.0) == 1.0


def test_cube():
    assert cube_activation(-1.0) == -1.0
    assert cube_activation(-0.5) == -0.125
    assert cube_activation(0.0) == 0.0
    assert cube_activation(0.5) == 0.125
    assert cube_activation(1.0) == 1.0
