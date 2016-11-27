from neat import activations


# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

class NotAlmostEqualException(Exception):
    pass


def assert_almost_equal(a, b):
    if abs(a - b) > 1e-6:
        max_abs = max(abs(a), abs(b))
        abs_rel_err = abs(a - b) / max_abs
        if abs_rel_err > 1e-6:
            raise NotAlmostEqualException("{0:.4f} !~= {1:.4f}".format(a, b))


def test_sigmoid():
    assert activations.sigmoid_activation(0.0) == 0.5


def test_tanh():
    assert activations.tanh_activation(0.0) == 0.0


def test_sin():
    assert activations.sin_activation(0.0) == 0.0


def test_gauss():
    assert_almost_equal(activations.gauss_activation(0.0), 1.0)
    assert_almost_equal(activations.gauss_activation(-1.0),
                        activations.gauss_activation(1.0))


def test_relu():
    assert activations.relu_activation(-1.0) == 0.0
    assert activations.relu_activation(0.0) == 0.0
    assert activations.relu_activation(1.0) == 1.0


def test_identity():
    assert activations.identity_activation(-1.0) == -1.0
    assert activations.identity_activation(0.0) == 0.0
    assert activations.identity_activation(1.0) == 1.0


def test_clamped():
    assert activations.clamped_activation(-2.0) == -1.0
    assert activations.clamped_activation(-1.0) == -1.0
    assert activations.clamped_activation(0.0) == 0.0
    assert activations.clamped_activation(1.0) == 1.0
    assert activations.clamped_activation(2.0) == 1.0


def test_inv():
    assert activations.inv_activation(1.0) == 1.0
    assert activations.inv_activation(0.5) == 2.0
    assert activations.inv_activation(2.0) == 0.5
    assert activations.inv_activation(0.0) == 0.0


def test_log():
    assert activations.log_activation(1.0) == 0.0


def test_exp():
    assert activations.exp_activation(0.0) == 1.0


def test_abs():
    assert activations.abs_activation(-1.0) == 1.0
    assert activations.abs_activation(0.0) == 0.0
    assert activations.abs_activation(-1.0) == 1.0


def test_hat():
    assert activations.hat_activation(-1.0) == 0.0
    assert activations.hat_activation(0.0) == 1.0
    assert activations.hat_activation(1.0) == 0.0


def test_square():
    assert activations.square_activation(-1.0) == 1.0
    assert activations.square_activation(-0.5) == 0.25
    assert activations.square_activation(0.0) == 0.0
    assert activations.square_activation(0.5) == 0.25
    assert activations.square_activation(1.0) == 1.0


def test_cube():
    assert activations.cube_activation(-1.0) == -1.0
    assert activations.cube_activation(-0.5) == -0.125
    assert activations.cube_activation(0.0) == 0.0
    assert activations.cube_activation(0.5) == 0.125
    assert activations.cube_activation(1.0) == 1.0


def test_function_set():
    s = activations.ActivationFunctionSet()
    assert s.get('sigmoid') is not None
    assert s.get('tanh') is not None
    assert s.get('sin') is not None
    assert s.get('gauss') is not None
    assert s.get('relu') is not None
    assert s.get('identity') is not None
    assert s.get('clamped') is not None
    assert s.get('inv') is not None
    assert s.get('log') is not None
    assert s.get('exp') is not None
    assert s.get('abs') is not None
    assert s.get('hat') is not None
    assert s.get('square') is not None
    assert s.get('cube') is not None

    assert s.is_valid('sigmoid')
    assert s.is_valid('tanh')
    assert s.is_valid('sin')
    assert s.is_valid('gauss')
    assert s.is_valid('relu')
    assert s.is_valid('identity')
    assert s.is_valid('clamped')
    assert s.is_valid('inv')
    assert s.is_valid('log')
    assert s.is_valid('exp')
    assert s.is_valid('abs')
    assert s.is_valid('hat')
    assert s.is_valid('square')
    assert s.is_valid('cube')

    assert not s.is_valid('foo')


if __name__ == '__main__':
    test_sigmoid()
    test_tanh()
    test_sin()
    test_gauss()
    test_relu()
    test_identity()
    test_clamped()
    test_inv()
    test_log()
    test_exp()
    test_abs()
    test_hat()
    test_square()
    test_cube()

