#ifndef NEURON_HPP
#define NEURON_HPP

#include <Python.h>

struct NeuronObject {
	PyObject_HEAD
	int id;
	PyListObject* synapses;
	double bias;
	const char* type;
	double response;
	double output; 
};

struct SynapseObject;

namespace Neuron {

PyObject* set_nn_activation(PyObject *self, PyObject *args);

double sigmoid(double x, double response);

int Neuron_init(NeuronObject *self, PyObject *args, PyObject *kwds);

PyObject* Neuron_get_type(NeuronObject* self, void* closure);

PyObject* Neuron_get_id(NeuronObject* self, void* closure);

PyObject* Neuron_get_output(NeuronObject* self, void* closure);

int Neuron_set_output(NeuronObject* self, PyObject *value, void *closure);

double update_activation(PyListObject* synapses);

PyObject* Neuron_activate(NeuronObject* self);

void create_synapse(NeuronObject* self, SynapseObject* s);

void Neuron_dealloc(NeuronObject* self);

extern PyTypeObject NeuronType;

}

#endif
