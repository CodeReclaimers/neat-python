// ******************************************************************//
// A class to handle general neural networks in matrix form.         //
// ------------------------------------------------------------------//
// To compile:
// g++ -I /usr/include/python2.5/ -c ANN.cpp
// g++ -lpython2.5 ANN.o -o ann.out

#ifndef _ANN_H_
#define _ANN_H_

#include <Python.h>

class ANN {
    public:
        ANN(int inputs, int neurons);
        //~ANN();

        void set_synapse(int from, int to, double value) {
            weights[from][to] = value; };

        void set_sensory_weight(int from, int to, double value) {sensory_weights[from][to] = value;};

        void set_neuron(int i, double bias, double gain, int type) {
            biases[i] = bias;
            response[i] = gain;
            neuron_type[i] = type;
        };

        double get_neuron_response(int i) { return response[i]; }
        double get_neuron_bias(int i) { return biases[i]; }

        void set_neuron_output(int i, double output) {
            outputs[i] = output;  };

        double get_neuron_output(int i) { return outputs[i]; };

        // serial activation method (for feedforward topologies)
        PyObject* sactivate(PyObject* inputs);
        // parallel activation method (for recurrent neural networks)
        PyObject* pactivate(PyObject* inputs);

        // flushes all neuron's output
        void flush();

        void set_logistic(bool option) {
            logistic = option;
        };

        double sigmoid(double x, double response);

       // void use_fast_sigmoid(bool b) {
       //     fast_sigmoid = b;
       // }

   private:
        int size;      // number of neurons (hidden + output)
        int sensors;   // number of sensors (inputs)
        bool logistic; // activation type (exp or tanh)

        // neuron's properties
        double *states, *outputs, *biases, *response;

        // each neuron has a type: 0 (hidden) or 1 (output)
        int *neuron_type;

        // each network is composed of two matrices:
        // one to specify how the inputs are connected
        // to neurons and other to specify inter-neuron
        // connections (hidden and outputs)
        double **weights, **sensory_weights;

};
#endif
