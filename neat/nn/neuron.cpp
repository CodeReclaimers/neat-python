#include "neuron.hpp"
#include "synapse.hpp"
#include <cstring>
#include <cmath>

namespace Neuron {

PyGetSetDef Neuron_getsetters[] = {
		{"id", reinterpret_cast<getter>(Neuron_get_id), 0,
				"Returns neuron's id", 0},
		{"type", reinterpret_cast<getter>(Neuron_get_type), 0,
				"Returns neuron's type: INPUT, OUTPUT, or HIDDEN", 0},
		{"output", reinterpret_cast<getter>(Neuron_get_output),
			reinterpret_cast<setter>(Neuron_set_output), "Output", 0},
     	{0}
};

const char* nn_activation;

PyObject* set_nn_activation(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "s", &nn_activation))
        return 0;
    return Py_BuildValue("");
}

// Sigmoidal type of activation function
double sigmoid(double x, double response) {
	if (std::strcmp(nn_activation, "exp") == 0) {
		if (x < - 30)
			return 0.0;
		else if (x > 30)
			return 1;
		else
			return 1.0 / (1.0 + std::exp(-x * response));
	}
    else {
    	if (x < - 20)
			return -1.0;
		else if (x > 20)
			return +1.0;
		else
			return std::tanh(x * response);
    }
}

int global_id;

int Neuron_init(NeuronObject *self, PyObject *args, PyObject *kwds) {
	self->id = 0;
    self->synapses = reinterpret_cast<PyListObject*>(PyList_New(0));
    if (!self->synapses) {
    	return -1;
    }
    self->bias = 0;
    self->response = 1;
    self->output = 0;
	
	static char *kwlist[] = {"neurontype", "id", "bias", "response", 0};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|idd", kwlist, 
    		&self->type, &self->id, &self->bias, &self->response)) {
        return -1;
    }
    
    if (!self->id) {
    	global_id++;
    	self->id = global_id;
    }
    return 0;
}

PyObject* Neuron_get_type(NeuronObject* self, void* closure)
{
    return Py_BuildValue("s", self->type);
}

PyObject* Neuron_get_id(NeuronObject* self, void* closure)
{
    return Py_BuildValue("i", self->id);
}

PyObject* Neuron_get_output(NeuronObject* self, void* closure)
{
    return Py_BuildValue("d", self->output);
}

int Neuron_set_output(NeuronObject* self, PyObject *value, void *closure)
{
	self->output = PyFloat_AsDouble(value);
	if (PyErr_Occurred()) {
		return -1;
	}
	return 0;
}

double update_activation(PyListObject* synapses)
{
    double soma = 0;
    for(int i = 0; i < PyList_GET_SIZE(synapses); i++) {
    	SynapseObject* s =
    		reinterpret_cast<SynapseObject*>(PyList_GET_ITEM(synapses, i));
        soma += Synapse::incoming(s);
    }
    return soma;
}

PyObject* Neuron_activate(NeuronObject* self)
{
    // Activates the neuron
	if (PyList_GET_SIZE(self->synapses)) {           
       return Py_BuildValue("d", sigmoid(update_activation(self->synapses) +
    		   self->bias, self->response));
    }
	else {
		return Py_BuildValue("d", self->output); // for input neurons (sensors)
	}
}

void create_synapse(NeuronObject* self, SynapseObject* s)
{ 
	Py_INCREF(s);
	PyList_Append(reinterpret_cast<PyObject*>(self->synapses),
			reinterpret_cast<PyObject*>(s));
}

void Neuron_dealloc(NeuronObject* self)
{
	for(int i = 0; i < PyList_GET_SIZE(self->synapses); i++) {
		Py_DECREF(PyList_GET_ITEM(self->synapses, i));
	}
    self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}

PyMethodDef Neuron_methods[] = {
    {"activate", reinterpret_cast<PyCFunction>(Neuron_activate), METH_NOARGS,
     "Activates the neuron"},
    {0}
};

PyTypeObject NeuronType = {
		PyObject_HEAD_INIT(0)
		0,							/* ob_size */
		"nn.Neuron",			    /* tp_name */
		sizeof(NeuronObject),		/* tp_basicsize */
		0,							/* tp_itemsize */
		reinterpret_cast<destructor>(Neuron_dealloc), /* tp_dealloc */
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
		"A simple sigmoidal neuron",
		0,		               		/* tp_traverse */
		0,		               		/* tp_clear */
		0,		               		/* tp_richcompare */
		0,		               		/* tp_weaklistoffset */
		0,		               		/* tp_iter */
		0,		               		/* tp_iternext */
		Neuron_methods,				/* tp_methods */
		0,             				/* tp_members */
		Neuron_getsetters,          /* tp_getset */
		0,                         /* tp_base */
		0,                         /* tp_dict */
		0,                         /* tp_descr_get */
		0,                         /* tp_descr_set */
		0,                         /* tp_dictoffset */
		reinterpret_cast<initproc>(Neuron_init),	/* tp_init */
};

}
