/*  dpole.cpp
 *
 *  C -> python interface to solve Wieland's equations of motion for the
 *  double pole balancing problem using the 4th order Runge-Kutta method.
 *
 *  A direct ripoff from Stanley's C++ code (with small changes) written
 *  by Richard Sutton, Charles Anderson, and Faustino Gomez.
 *
 *  This code is intended to be used with neat-python:
 *  http://code.google.com/p/neat-python
 *
 *  To compile manually:
 *  g++ -O2 -fPIC -Wall -I /usr/include/python2.5 -c dpole.cpp -o dpole.o
 *  g++ -lpython2.5 -shared dpole.o -o dpole.so
 *
 *  See setup.py for instructions on getting distutils to compile for you
 *  (odds are you just need to do 'python setup.py build_ext -i').
*/

#include <Python.h>
#include <vector>
#include <cmath>

using namespace std;

const double FORCE_MAG  = 10.0;
const double GRAVITY    = -9.8;
const double LENGTH_1   = 0.5;
const double LENGTH_2   = 0.05;
const double MASSPOLE_1 = 0.1;
const double MASSPOLE_2 = 0.01;
const double MASSCART   = 1.0;
const double MUP        = 0.000002;

void step(double action, const vector<double> &state, vector<double> &dydx) {

    double force, costheta_1, costheta_2, sintheta_1, sintheta_2;
    double gsintheta_1, gsintheta_2, temp_1, temp_2;
    double ml_1, ml_2, fi_1, fi_2, mi_1, mi_2;

    force       =  (action - 0.5) * FORCE_MAG * 2.0;
    costheta_1  = cos(state[2]);
    sintheta_1  = sin(state[2]);
    gsintheta_1 = GRAVITY*sintheta_1;
    costheta_2  = cos(state[4]);
    sintheta_2  = sin(state[4]);
    gsintheta_2 = GRAVITY*sintheta_2;

    ml_1   = LENGTH_1 * MASSPOLE_1;
    ml_2   = LENGTH_2 * MASSPOLE_2;
    temp_1 = MUP * state[3] / ml_1;
    temp_2 = MUP * state[5] / ml_2;

    fi_1 = (ml_1 * state[3] * state[3] * sintheta_1) +
           (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
    fi_2 = (ml_2 * state[5] * state[5] * sintheta_2) +
           (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));

    mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));
    mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));

    dydx[1] = (force + fi_1 + fi_2) / (mi_1 + mi_2 + MASSCART);

    dydx[3] = -0.75 * (dydx[1] * costheta_1 + gsintheta_1 + temp_1) / LENGTH_1;
    dydx[5] = -0.75 * (dydx[1] * costheta_2 + gsintheta_2 + temp_2) / LENGTH_2;
}

void rk4(double f, vector<double> &state, const vector<double> &dydx) {

    int i;
    double TAU = 0.01;
    double hh = TAU*0.5;
    double h6 = TAU/6.0;
    vector<double> dym(6), dyt(6), yt(6);

    for(i = 0; i < 6; i++)
        yt[i] = state[i] + hh*dydx[i];

    step(f, yt, dyt);

    dyt[0] = yt[1];
    dyt[2] = yt[3];
    dyt[4] = yt[5];

    for(i = 0; i < 6; i++)
        yt[i] = state[i] + hh * dyt[i];

    step(f, yt, dym);

    dym[0] = yt[1];
    dym[2] = yt[3];
    dym[4] = yt[5];

    for(i = 0; i < 6; i++) {
        yt[i] = state[i] + TAU * dym[i];
        dym[i] += dyt[i];
    }

    step(f, yt, dyt);

    dyt[0] = yt[1];
    dyt[2] = yt[3];
    dyt[4] = yt[5];

    for(i = 0; i < 6; i++)
        state[i] += h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
}

bool list2vector(PyObject* list, vector<double>& v) {
    for(int i = 0; i < 6; i++) {
        v[i] = PyFloat_AsDouble(PyList_GET_ITEM(list, i));
        if (PyErr_Occurred()) return false;
    }
    return true;
}

PyObject* vector2list(const vector<double>& v) {
    PyObject* list = PyList_New(6);
    if (!list) return 0;
    for(int i = 0; i < 6; i++) {
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(v[i]));
    }
    return list;
}

PyObject* integrate(PyObject* self, PyObject* args) {

    PyObject* list;
    int stepnum;
    double output;
    vector<double> state(6), dydx(6);

    if (!PyArg_ParseTuple(args, "dO!i", &output, &PyList_Type, &list, &stepnum))
        return 0;

    Py_INCREF(list);
    if(!list2vector(list, state)) {
        Py_DECREF(list);
        return 0;
        }
    else {
       Py_DECREF(list);
    }

    /*--- Apply action to the simulated cart-pole ---*/
    for(int k = 0; k < stepnum; k++) {
        for(int i = 0; i < 2; i++){
            dydx[0] = state[1];
            dydx[2] = state[3];
            dydx[4] = state[5];
            step(output, state, dydx);
            rk4(output, state, dydx);
        }
    }

    return vector2list(state);
}


static PyMethodDef functions[] = {
    {"integrate", integrate, METH_VARARGS},
    {NULL, NULL, 0}
};

PyMODINIT_FUNC initdpole(void)
{
    Py_InitModule3("dpole", functions, "Integrate double cart-pole equations.");
}
