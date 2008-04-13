// ******************************************************************//
// A class to handle general neural networks in matrix form.         //
// ------------------------------------------------------------------//
// The code was specifically designed to be called from neat-python  //
// and was partially based on the source code provided by Randall    //
// D. Beer at mypage.iu.edu/~rdbeer/                                 //
// ******************************************************************//

#include "ANN.h"
#include <stdlib.h>

// ****************************
// Constructors and Destructors
// ****************************

// The constructor

ANN::ANN(int how_many_inputs, int newsize) {
    sensors = how_many_inputs;
    size = newsize;

    weights.SetBounds(0,size-1,0,size-1);
    weights.FillContents(0.0);
    
    logistic = true;
    
    sensory_weights.SetBounds(0,sensors-1,0,size-1);
    sensory_weights.FillContents(0.0);
    
    states.SetBounds(0,size-1);
    states.FillContents(0.0);
    
    outputs.SetBounds(0,size-1);
    outputs.FillContents(0.0);

    biases.SetBounds(0,size-1);
    biases.FillContents(0.0);

    response.SetBounds(0,size-1);
    response.FillContents(1.0);

    neuron_type.SetBounds(0,size-1);
    neuron_type.FillContents(0);
}

// flushes all neuron's output
void ANN::flush() {
    for (int i = 0; i < size; i++) {
        outputs[i] = 0.0;
    }
}

// serial activation method (for feedforward topologies)
PyObject* ANN::sactivate(PyObject* inputs)
{
	if (PyList_Size(inputs) != sensors) {
    	PyErr_SetString(PyExc_ValueError, "Wrong number of inputs.");
    	return 0;
    }
    
    PyObject* output = PyList_New(0);
    if (!output) return 0;

    // Update the state of all neurons.
    for (int i = 0; i < size; i++) {
        double neuron_input = 0.0;

        // inputs from the outside (sensors)
        for (int j = 0; j < sensors; j++) {
            PyObject* inputj = PyList_GetItem(inputs, j);
            if (!inputj) {
                Py_DECREF(output);
                return 0;
            }
            double inputjd = PyFloat_AsDouble(inputj);
            if (PyErr_Occurred()) {
                Py_DECREF(output);
                return 0;
            }
            neuron_input += sensory_weights[j][i] * inputjd;
        }
        // signal coming from other neurons
        for (int j = 0; j < size; j++)
            neuron_input += weights[j][i] * outputs[j];

        states[i]  = neuron_input;

        outputs[i] = sigmoid(states[i] + biases[i], response[i]);

        if(neuron_type[i] == 1) {
            if (PyList_Append(output, PyFloat_FromDouble(outputs[i])) != 0) {
                Py_DECREF(output);
                return 0;
            }
        }
    }
    return output;
}

// parallel activation method (for recurrent neural networks)
PyObject*  ANN::pactivate(PyObject* inputs) {
	if (PyList_Size(inputs) != sensors) {
    	PyErr_SetString(PyExc_ValueError, "Wrong number of inputs.");
    	return 0;
    }

    PyObject* output = PyList_New(0);
    if (!output) return 0;

    // Update the state of all neurons.
    for (int i = 0; i < size; i++) {
        double neuron_input = 0;

        // inputs from the outside (sensors)
        for (int j = 0; j < sensors; j++) {
            PyObject* inputj = PyList_GetItem(inputs, j);
            if (!inputj) {
                Py_DECREF(output);
                return 0;
            }
            double inputjd = PyFloat_AsDouble(inputj);
            if (PyErr_Occurred()) {
                Py_DECREF(output);
                return 0;
            }
            neuron_input += sensory_weights[j][i] * inputjd;
        }

        // signal coming from other neurons
        for (int j = 0; j < size; j++)
            neuron_input += weights[j][i] * outputs[j];

        states[i] = neuron_input;
    }

    for (int i = 0; i < size; i++) {
        outputs[i] = sigmoid(states[i] + biases[i], response[i]);

        if(neuron_type[i] == 1) {
            if (PyList_Append(output, PyFloat_FromDouble(outputs[i])) != 0) {
                Py_DECREF(output);
                return 0;
            }
        }
    }
    return output;
}

// The sigmoid function
double ANN::sigmoid(double x, double response) {
    if(logistic) {
        if (x < - 30.0) 
            return 0.0;
        else if (x > 30.0) 
            return 1.0;
        else 
            return 1.0/(1.0 + exp(-x*response));
    }
    else {
        if (x < - 20.0) 
            return -1.0;
        else if (x > 20.0) 
            return 1.0;
        else 
            return tanh(x*response);
    }
}


