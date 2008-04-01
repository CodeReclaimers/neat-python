#include <Python.h>
//#include "nn.hpp"
#include "neuron.hpp"
#include "synapse.hpp"

namespace {

PyMethodDef methods[] = {
		{"set_nn_activation",  Neuron::set_nn_activation, METH_VARARGS, ""},
		{0}
};

}

PyMODINIT_FUNC initnn_cpp(void)
{	
	PyObject* module = Py_InitModule("nn_cpp", methods);
	
	/* Neuron */
	
	Neuron::NeuronType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&Neuron::NeuronType) < 0)
		return;
	
	Py_INCREF(&Neuron::NeuronType);
	PyModule_AddObject(module, "Neuron", reinterpret_cast<PyObject*>(&Neuron::NeuronType));
	
	/* Synapse */
		
	Synapse::SynapseType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&Synapse::SynapseType) < 0)
		return;
		
	Py_INCREF(&Synapse::SynapseType);
	PyModule_AddObject(module, "Synapse", reinterpret_cast<PyObject*>(&Synapse::SynapseType));
}
