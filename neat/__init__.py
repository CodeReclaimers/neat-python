from neat import activations

activation_functions = activations.ActivationFunctionSet()

activation_functions.add('sigmoid', activations.sigmoid_activation)
activation_functions.add('tanh', activations.tanh_activation)
activation_functions.add('sin', activations.sin_activation)
activation_functions.add('gauss', activations.gauss_activation)
activation_functions.add('relu', activations.relu_activation)
activation_functions.add('identity', activations.identity_activation)
activation_functions.add('clamped', activations.clamped_activation)
activation_functions.add('inv', activations.inv_activation)
activation_functions.add('log', activations.log_activation)
activation_functions.add('exp', activations.exp_activation)
activation_functions.add('abs', activations.abs_activation)
activation_functions.add('hat', activations.hat_activation)
activation_functions.add('square', activations.square_activation)
activation_functions.add('cube', activations.cube_activation)
