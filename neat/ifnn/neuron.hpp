#ifndef NEURON_HPP
#define NEURON_HPP

struct NeuronObject {
	PyObject_HEAD
	double invtau;
	double vrest;
	double vreset;
	double vt;
	long double v;
	bool has_fired;
	double bias;
	double current;
};

namespace {

const double BIAS = 0;
const double TAU = 10;
const double VREST = -70;
const double VRESET = -70;
const double VT = -55;

int Neuron_init(NeuronObject *self, PyObject *args, PyObject *kwds) {
	self->bias = BIAS;
	double tau = TAU;
	self->vrest = VREST;
	self->vreset = VRESET;
	self->vt = VT;
	
	static char *kwlist[] = {"bias", "tau", "vrest", "vreset", "vt", 0};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ddddd", kwlist, 
    		&self->bias, &tau, &self->vrest, &self->vreset, &self->vt)) {
        return -1;
    }
    self->invtau = 1 / tau;
    self->v = self->vreset;
    self->has_fired = false;
    self->current = self->bias;
    return 0;
}

PyObject* Neuron_get_potential(NeuronObject *self, void *closure)
{
    return Py_BuildValue("d", double(self->v));
}

PyObject* Neuron_get_has_fired(NeuronObject* self, void* closure)
{
	if (self->has_fired) {
		Py_INCREF(Py_True);
		return Py_True;
	}
	else {
		Py_INCREF (Py_False);
		return Py_False;
	}
}

PyObject* Neuron_get_current(NeuronObject* self, void* closure)
{
	return Py_BuildValue("d", self->current);
}

int Neuron_set_current(NeuronObject *self, PyObject *value, void *closure)
{
	self->current = PyFloat_AsDouble(value);
	if (PyErr_Occurred()) {
		return -1;
	}
	return 0;
}


PyGetSetDef Neuron_getsetters[] = {
		{"potential", reinterpret_cast<getter>(Neuron_get_potential), 0,
			"Membrane potential", 0},
		{"has_fired", reinterpret_cast<getter>(Neuron_get_has_fired), 0,
			"Indicates whether the neuron has fired", 0},
		{"current", reinterpret_cast<getter>(Neuron_get_current),
			reinterpret_cast<setter>(Neuron_set_current), "current", 0},
     	{0}
};

PyObject* Neuron_advance(NeuronObject* self) {
	self->v += self->invtau * (self->vrest - self->v + self->current);
	if (self->v >= self->vt) {
		self->has_fired = true;
	    self->v = self->vreset;
	}
	else {
		self->has_fired = false;
	}
	self->current = self->bias;
	return Py_BuildValue("");
}

PyObject* Neuron_reset(NeuronObject* self) {
	self->v = self->vreset;
	self->has_fired = false;
	self->current = self->bias;
	return Py_BuildValue("");
}

PyMethodDef Neuron_methods[] = {
    {"advance", reinterpret_cast<PyCFunction>(Neuron_advance), METH_NOARGS,
    	"Advances time in 1 ms."},
    {"reset", reinterpret_cast<PyCFunction>(Neuron_reset), METH_NOARGS,
    	"Resets all state variables."},
    {0}
};

PyTypeObject NeuronType = {
		PyObject_HEAD_INIT(0)
		0,							/* ob_size */
		"ifnn.Neuron",			/* tp_name */
		sizeof(NeuronObject),		/* tp_basicsize */
		0,							/* tp_itemsize */
		0,							/* tp_dealloc */
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
		"Neuron based on the integrate and fire model",
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

#endif
