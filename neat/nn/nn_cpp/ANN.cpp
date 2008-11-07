// ******************************************************************//
// A class to handle general neural networks in matrix form.         //
// ------------------------------------------------------------------//
// To compile:
// g++ -I /usr/include/python2.5/ -c ANN.cpp
// g++ -lpython2.5 ANN.o -o ann.out

#include <iostream>
#include "ANN.h"

ANN::ANN(int inputs, int neurons) {

    sensors = inputs;
    size = neurons;
    logistic = true;

    sensory_weights = new double*[sensors];
    weights = new double*[size];

    for(int i=0; i<sensors; i++)
        sensory_weights[i] = new double[size];

    for(int i=0; i<size; i++)
        weights[i] = new double[size];

    states   = new double[size];
    outputs  = new double[size];
    biases   = new double[size];
    response = new double[size];

    neuron_type = new int[size];

    // set everything to zero
    for(int i=0; i<size; i++) {
        states[i]   = 0.0;
        outputs[i]  = 0.0;
        biases[i]   = 0.0;
        response[i] = 0.0;
        neuron_type[i] = 0;
    }

    for(int i=0; i<sensors; i++) {
        for(int j=0; j<size; j++) {
            sensory_weights[i][j] = 0.0;
        }
    }

    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            weights[i][j] = 0.0;
        }
    }

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

    double neuron_input, inputjd;
    PyObject* inputj;

    // Update the state of all neurons.
    for (int i = 0; i < size; i++) {
        neuron_input = 0.0;

        // inputs from the outside (sensors)
        for (int j = 0; j < sensors; j++) {
            inputj = PyList_GetItem(inputs, j);

            if (!inputj) {
                Py_DECREF(output);
                return 0;
            }

            inputjd = PyFloat_AsDouble(inputj);

            // pode dar problemas ao rodar a main()
            //if (PyErr_Occurred()) {
            //    Py_DECREF(output);
            //    return 0;
            //}

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

    double neuron_input, inputjd;
    PyObject* inputj;

    // Update the state of all neurons.
    for (int i = 0; i < size; i++) {
        neuron_input = 0.0;

        // inputs from the outside (sensors)
        for (int j = 0; j < sensors; j++) {
            inputj = PyList_GetItem(inputs, j);
            if (!inputj) {
                Py_DECREF(output);
                return 0;
            }

            inputjd = PyFloat_AsDouble(inputj);

            // pode dar problemas ao rodar a main()
            //if (PyErr_Occurred()) {
            //    Py_DECREF(output);
            //    return 0;
           // }

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
inline double ANN::sigmoid(double x, double response) {
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

// to compile manually:
// g++ -I /usr/include/python2.5/ -lpython2.5 ANN.cpp
int main() {

    ANN* rede = new ANN(3,3);

    // bias input
    rede->set_sensory_weight(0, 0, 1.5);
    rede->set_sensory_weight(0, 1, 1.5);
    rede->set_sensory_weight(0, 2, 1.5);
    // input 1
    rede->set_sensory_weight(1, 0, 1.5);
    rede->set_sensory_weight(1, 1, 1.5);
    // input 2
    rede->set_sensory_weight(2, 0, 1.5);
    rede->set_sensory_weight(2, 1, 1.5);
    // inter-neurons
    rede->set_synapse(0, 2, 0.5);
    rede->set_synapse(1, 2, 0.5);
    rede->set_synapse(2, 1, -0.5);

    // neuron's properties: id, bias, response, type
    rede->set_neuron(0, 0, 1, 0); // hidden
    rede->set_neuron(1, 0, 1, 0); // hidden
    rede->set_neuron(2, 0, 1, 1); // output

    // input
    PyObject* inputs = PyList_New(0);
    PyList_Append(inputs, PyFloat_FromDouble(1.2)); //bias
    PyList_Append(inputs, PyFloat_FromDouble(0.2));
    PyList_Append(inputs, PyFloat_FromDouble(0.2));

    // activate
    std::cout << "Serial activation..." << std::endl;
    PyObject* output;
    for(int i=0; i<10; i++) {
        output = rede->sactivate(inputs);
        std::cout << "Output: " << PyFloat_AsDouble(PyList_GetItem(output, 0)) << std::endl;
    }
    
    //std::cout << rede->get_neuron_output(0) << std::endl;
    //std::cout << rede->get_neuron_output(1) << std::endl;
    //std::cout << rede->get_neuron_output(2) << std::endl;

    return 0;
}

