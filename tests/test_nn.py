from neat import nn

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
    assert nn.sigmoid_activation(0.0, 1.0, 0.0) == 0.5


def test_tanh():
    assert nn.tanh_activation(0.0, 1.0, 0.0) == 0.0


def test_sin():
    assert nn.sin_activation(0.0, 1.0, 0.0) == 0.0


def test_gauss():
    assert_almost_equal(nn.gauss_activation(0.0, 1.0, 0.0), 0.398942280401)
    assert_almost_equal(nn.gauss_activation(0.0, 1.0, -1.0),
                        nn.gauss_activation(0.0, 1.0, 1.0))


def test_relu():
    assert nn.relu_activation(0.0, 1.0, -1.0) == 0.0
    assert nn.relu_activation(0.0, 1.0, 0.0) == 0.0
    assert nn.relu_activation(0.0, 1.0, 1.0) == 1.0


def test_identity():
    assert nn.identity_activation(0.0, 1.0, -1.0) == -1.0
    assert nn.identity_activation(0.0, 1.0, 0.0) == 0.0
    assert nn.identity_activation(0.0, 1.0, 1.0) == 1.0


def test_clamped():
    assert nn.clamped_activation(0.0, 1.0, -2.0) == -1.0
    assert nn.clamped_activation(0.0, 1.0, -1.0) == -1.0
    assert nn.clamped_activation(0.0, 1.0, 0.0) == 0.0
    assert nn.clamped_activation(0.0, 1.0, 1.0) == 1.0
    assert nn.clamped_activation(0.0, 1.0, 2.0) == 1.0


def test_inv():
    assert nn.inv_activation(0.0, 1.0, 1.0) == 1.0


def test_log():
    assert nn.log_activation(0.0, 1.0, 1.0) == 0.0


def test_exp():
    assert nn.exp_activation(0.0, 1.0, 0.0) == 1.0


def test_abs():
    assert nn.abs_activation(0.0, 1.0, -1.0) == 1.0
    assert nn.abs_activation(0.0, 1.0, 0.0) == 0.0
    assert nn.abs_activation(0.0, 1.0, -1.0) == 1.0


def test_hat():
    assert nn.hat_activation(0.0, 1.0, -1.0) == 0.0
    assert nn.hat_activation(0.0, 1.0, 0.0) == 1.0
    assert nn.hat_activation(0.0, 1.0, 1.0) == 0.0
