#include "synapse.hpp"
#include "neuron.hpp"

namespace Synapse {

int Synapse_init(SynapseObject *self, PyObject *args, PyObject *kwds) {
	static char *kwlist[] = {"source", "dest", "weight", 0};
	PyObject* source = 0;
	PyObject* destination = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!d", kwlist, 
    		&Neuron::NeuronType, &source, &Neuron::NeuronType, &destination,
    		&self->weight)) {
        return -1;
    }
    Py_INCREF(source);
    self->source = reinterpret_cast<NeuronObject*>(source);
    Py_INCREF(destination);
    self->destination = reinterpret_cast<NeuronObject*>(destination);
    Neuron::create_synapse(self->destination, self);
    return 0;
}

double incoming(SynapseObject *self)
{
	// Receives the incoming signal from a sensor or another neuron
	// and returns the value to the neuron it belongs to.
	return self->weight * self->source->output;
}

void Synapse_dealloc(SynapseObject* self)
{
    Py_XDECREF(self->source);
    Py_XDECREF(self->destination);
    self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}

PyMethodDef Synapse_methods[] = {
    {0}
};

PyTypeObject SynapseType = {
		PyObject_HEAD_INIT(0)
		0,							/* ob_size */
		"nn.Synapse",		/* tp_name */
		sizeof(SynapseObject),		/* tp_basicsize */
		0,							/* tp_itemsize */
		reinterpret_cast<destructor>(Synapse_dealloc),	/* tp_dealloc */
		0,							/* tp_print */
		0,							/* tp_getattr */
		0,							/* tp_setattr */
		0,							/* tp_compare */
		0,							/* tp_repr */
		0,							/* tp_as_number */
		0,							/* tp_as_sequence */
		0,							/* tp_as_mapping */
		0,							/* tp_hash */
		0,							/* tp_call */
		0,							/* tp_str */
		0,							/* tp_getattro */
		0,							/* tp_setattro */
		0,							/* tp_as_buffer */
		Py_TPFLAGS_DEFAULT,			/* tp_flags */
		"A synapse indicates the connection strength between two neurons (or itself)",
		0,		               		/* tp_traverse */
		0,		               		/* tp_clear */
		0,		               		/* tp_richcompare */
		0,		               		/* tp_weaklistoffset */
		0,		               		/* tp_iter */
		0,		               		/* tp_iternext */
		Synapse_methods,			/* tp_methods */
		0,             				/* tp_members */
		0, 							/* tp_getset */
		0,                         /* tp_base */
		0,                         /* tp_dict */
		0,                         /* tp_descr_get */
		0,                         /* tp_descr_set */
		0,                         /* tp_dictoffset */
		reinterpret_cast<initproc>(Synapse_init),	/* tp_init */
};

}
