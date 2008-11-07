//*****************************************************//
// An interface to wrap the C++ neural network class   //
// into a Python shared library.                       //
//*****************************************************//
#include <Python.h>
#include "PyANN.hpp"

namespace {

PyMethodDef ANN_methods[] = {
    {"set_synapse", reinterpret_cast<PyCFunction>(set_synapse),
        METH_VARARGS, ""},
    {"set_sensory_weight", reinterpret_cast<PyCFunction>(set_sensory_weight),
        METH_VARARGS, ""},
    {"set_neuron", reinterpret_cast<PyCFunction>(set_neuron),
        METH_VARARGS, ""},
    {"get_neuron_response", reinterpret_cast<PyCFunction>(get_neuron_response),
        METH_VARARGS, ""},
    {"get_neuron_bias", reinterpret_cast<PyCFunction>(get_neuron_bias),
        METH_VARARGS, ""},
    {"set_neuron_output", reinterpret_cast<PyCFunction>(set_neuron_output),
        METH_VARARGS, ""},
    {"get_neuron_output", reinterpret_cast<PyCFunction>(get_neuron_output),
        METH_VARARGS, ""},
    {"sactivate", reinterpret_cast<PyCFunction>(sactivate),
        METH_VARARGS, ""},
    {"pactivate", reinterpret_cast<PyCFunction>(pactivate),
        METH_VARARGS, ""},
    {"flush", reinterpret_cast<PyCFunction>(flush),
        METH_NOARGS, ""},
    {"set_logistic", reinterpret_cast<PyCFunction>(set_logistic),
        METH_VARARGS, ""},
    {0}
};

PyTypeObject ANNType = {
        PyObject_HEAD_INIT(0)
        0,                            /* ob_size */
        "ann.ANN",                    /* tp_name */
        sizeof(ANNObject),            /* tp_basicsize */
        0,                            /* tp_itemsize */
        reinterpret_cast<destructor>(ANN_dealloc), /* tp_dealloc */
        0,                            /* tp_print */
        0,                            /* tp_getattr */
        0,                            /* tp_setattr */
        0,                            /* tp_compare */
        0,                            /* tp_repr */
        0,                            /* tp_as_number */
        0,                            /* tp_as_sequence */
        0,                            /* tp_as_mapping */
        0,                            /* tp_hash */
        0,                            /* tp_call */
        0,                            /* tp_str */
        0,                            /* tp_getattro */
        0,                            /* tp_setattro */
        0,                            /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,           /* tp_flags */
        "Classe para tratar redes neurais matricialmente",
        0,                            /* tp_traverse */
        0,                            /* tp_clear */
        0,                            /* tp_richcompare */
        0,                            /* tp_weaklistoffset */
        0,                            /* tp_iter */
        0,                            /* tp_iternext */
        ANN_methods,                  /* tp_methods */
        0,                            /* tp_members */
        0,                            /* tp_getset */
        0,                            /* tp_base */
        0,                            /* tp_dict */
        0,                            /* tp_descr_get */
        0,                            /* tp_descr_set */
        0,                            /* tp_dictoffset */
        reinterpret_cast<initproc>(ANN_init),    /* tp_init */
};

PyMethodDef methods[] = {
    {0}
};

}

PyMODINIT_FUNC initann(void)
{    
    PyObject* module = Py_InitModule("ann", methods);
    
    /* ANN */    
    ANNType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ANNType) < 0)
        return;
    
    Py_INCREF(&ANNType);
    PyModule_AddObject(module, "ANN", reinterpret_cast<PyObject*>(&ANNType));
}
