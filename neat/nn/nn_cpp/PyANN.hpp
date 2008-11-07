// **************************************************** //
// An interface to wrap the C++ neural network class    //
// into a Python shared library.                        //
// **************************************************** //
#ifndef _PYANN_HPP_
#define _PYANN_HPP_

#include <Python.h>
//#include <vector>
#include "ANN.h"

struct ANNObject {
    PyObject_HEAD
    ANN* ann;
};

namespace {

// constructor
int ANN_init(ANNObject *self, PyObject *args, PyObject *kwds) {
    int inputs;
    int neurons;
    static char *kwlist[] = {"inputs", "neurons", 0};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
            &inputs, &neurons)) {
        return -1;
    }
    self->ann = new ANN(inputs, neurons);
    return 0;
}
// destructor
void ANN_dealloc(ANNObject* self)
{
    delete self->ann;
    self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}

// methods
PyObject* set_synapse(ANNObject *self, PyObject *args) {
    int from, to;
    double value;
    if (!PyArg_ParseTuple(args, "iid", &from, &to, &value)) {
        return 0;
    }
    self->ann->set_synapse(from, to, value);
    return Py_BuildValue("");
}

PyObject* set_sensory_weight(ANNObject *self, PyObject *args) {
    int from, to;
    double value;
    if (!PyArg_ParseTuple(args, "iid", &from, &to, &value)) {
        return 0;
    }
    self->ann->set_sensory_weight(from, to, value);
    return Py_BuildValue("");
}

PyObject* set_neuron(ANNObject *self, PyObject *args) {
    int i, type;
    double bias, gain;
    if (!PyArg_ParseTuple(args, "iddi", &i, &bias, &gain, &type)) {
        return 0;
    }
    self->ann->set_neuron(i, bias, gain, type);
    return Py_BuildValue("");
}

PyObject* get_neuron_response(ANNObject *self, PyObject *args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) {
        return 0;
    }
    return Py_BuildValue("d", self->ann->get_neuron_response(i));
}

PyObject* get_neuron_bias(ANNObject *self, PyObject *args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) {
        return 0;
    }
    return Py_BuildValue("d", self->ann->get_neuron_bias(i));
}

PyObject* set_neuron_output(ANNObject *self, PyObject *args) {
    int i;
    double output;
    if (!PyArg_ParseTuple(args, "id", &i, &output)) {
        return 0;
    }
    self->ann->set_neuron_output(i, output);
    return Py_BuildValue("");
}

PyObject* get_neuron_output(ANNObject *self, PyObject *args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) {
        return 0;
    }
    return Py_BuildValue("d", self->ann->get_neuron_output(i));
}

PyObject* sactivate(ANNObject *self, PyObject *args) {
    PyObject* list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list)) {
        return 0;
    }
    Py_INCREF(list);
    PyObject* output = self->ann->sactivate(list);
    Py_DECREF(list);
    return output;
}

PyObject* pactivate(ANNObject *self, PyObject *args) {
    PyObject* list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list)) {
        return 0;
    }
    Py_INCREF(list);
    PyObject* output = self->ann->pactivate(list);
    Py_DECREF(list);
    return output;
}

PyObject* flush(ANNObject* self) {
    self->ann->flush();
    return Py_BuildValue("");
}

PyObject* set_logistic(ANNObject *self, PyObject *args) {
    int option;
    if (!PyArg_ParseTuple(args, "i", &option)) {
        return 0;
    }
    self->ann->set_logistic(bool(option));
    return Py_BuildValue("");
}

}

#endif
