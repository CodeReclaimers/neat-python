// **************************************************** //
// An interface to wrap the C++ neural network class    //
// into a Python shared library.                        //
// **************************************************** //
#ifndef ANN_PYTHON
#define ANN_PYTHON

#include <Python.h>
#include <vector>
#include "ANN.h"

struct ANNObject {
    PyObject_HEAD
    ANN* ann;
};

namespace {

// ****************************
// Constructor and destructor
// ****************************
int ANN_init(ANNObject *self, PyObject *args, PyObject *kwds) {
    int how_many_inputs = 0;
    int newsize = 0;
    static char *kwlist[] = {"how_many_inputs", "newsize", 0};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, 
            &how_many_inputs, &newsize)) {
        return -1;
    }
    self->ann = new ANN(how_many_inputs, newsize);
    return 0;
}

void ANN_dealloc(ANNObject* self)
{
    delete self->ann;
    self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}

// ****************************
// Methods
// ****************************
PyObject* SetConnectionWeight(ANNObject *self, PyObject *args) {
    int from, to;
    double value;
    if (!PyArg_ParseTuple(args, "iid", &from, &to, &value)) {
        return 0;
    }
    self->ann->SetConnectionWeight(from, to, value);
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

PyObject* setNeuronParameters(ANNObject *self, PyObject *args) {
    int i, type;
    double bias, gain;
    if (!PyArg_ParseTuple(args, "iddi", &i, &bias, &gain, &type)) {
        return 0;
    }
    self->ann->setNeuronParameters(i, bias, gain, type);
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

PyObject* setNeuronOutput(ANNObject *self, PyObject *args) {
    int i;
    double output;
    if (!PyArg_ParseTuple(args, "id", &i, &output)) {
        return 0;
    }
    self->ann->setNeuronOutput(i, output);
    return Py_BuildValue("");
}

PyObject* NeuronOutput(ANNObject *self, PyObject *args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) {
        return 0;
    }
    return Py_BuildValue("d", self->ann->NeuronOutput(i));
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
