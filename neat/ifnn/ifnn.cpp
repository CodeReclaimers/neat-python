#include <Python.h>
#include "neuron.hpp"
#include "synapse.hpp"

namespace {

PyMethodDef methods[] = {
		{0}
};

}

PyMODINIT_FUNC initifnn_cpp(void)
{	
	PyObject* module = Py_InitModule("ifnn_cpp", methods);
	
	/* Neuron */
	
	NeuronType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&NeuronType) < 0)
		return;
	
	Py_INCREF(&NeuronType);
	PyModule_AddObject(module, "Neuron", reinterpret_cast<PyObject*>(&NeuronType));
	
	/* Synapse */
		
	SynapseType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&SynapseType) < 0)
		return;
		
	Py_INCREF(&SynapseType);
	PyModule_AddObject(module, "Synapse", reinterpret_cast<PyObject*>(&SynapseType));
}
