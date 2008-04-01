#ifndef SYNAPSE_HPP
#define SYNAPSE_HPP

#include <Python.h>

struct NeuronObject;

struct SynapseObject {
	PyObject_HEAD
	double weight;
	NeuronObject* source;
	NeuronObject* destination;
};

namespace Synapse {

int Synapse_init(SynapseObject *self, PyObject *args, PyObject *kwds);

double incoming(SynapseObject *self);

void Synapse_dealloc(SynapseObject* self);

extern PyTypeObject SynapseType;

}

#endif
