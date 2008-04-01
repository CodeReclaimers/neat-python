/*  dpole_integrate.hpp
 * 
 *  Solves Wieland's equations of motion for the double pole
 *  balancing problem using Runge-Kutta forth-order method.
 * 
 *  A direct ripoff from Stanley's C++ code (with small changes) written 
 *  by Richard Sutton, Charles Anderson, and Faustino Gomez.
 * 
 *  This code is intended to be used with neat-python:
 *  http://code.google.com/p/neat-python    
*/
   
#include <vector>
#include <cmath>

using namespace std;
   
void step(double action, const vector<double> &st, vector<double> &derivs) {
    
    double FORCE_MAG = 10.0;
    double GRAVITY = -9.8;
    double LENGTH_1 = 0.5;
    double LENGTH_2 = 0.05;
    double MASSPOLE_1 = 0.1;
    double MASSPOLE_2 = 0.01;
    double MASSCART = 1.0;
    double MUP = 0.000002;    
    
    double force, costheta_1, costheta_2, sintheta_1, sintheta_2;
    double gsintheta_1, gsintheta_2, temp_1, temp_2;
    double ml_1, ml_2, fi_1, fi_2, mi_1, mi_2;
    
    force =  (action - 0.5) * FORCE_MAG * 2;
    costheta_1 = cos(st[2]);
    sintheta_1 = sin(st[2]);
    gsintheta_1 = GRAVITY * sintheta_1;
    costheta_2 = cos(st[4]);
    sintheta_2 = sin(st[4]);
    gsintheta_2 = GRAVITY * sintheta_2;
    
    ml_1 = LENGTH_1 * MASSPOLE_1;
    ml_2 = LENGTH_2 * MASSPOLE_2;
    temp_1 = MUP * st[3] / ml_1;
    temp_2 = MUP * st[5] / ml_2;
    
    fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) +
           (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
    fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) +
           (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
           
    mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));
    mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));
    
    derivs[1] = (force + fi_1 + fi_2)
                 / (mi_1 + mi_2 + MASSCART);
    
    derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1)
                 / LENGTH_1;
    derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2)
                  / LENGTH_2;
}

void rk4(double f, const vector<double> y, const vector<double> &dydx, vector<double> &yout) {

	int i;

	double hh,h6;
	
	vector<double> dym(6), dyt(6), yt(6);

    double TAU = 0.01;
    
	hh=TAU*0.5;
	h6=TAU/6.0;
	for (i=0;i<=5;i++) yt[i]=y[i]+hh*dydx[i];
	step(f,yt,dyt);
	dyt[0] = yt[1];
	dyt[2] = yt[3];
	dyt[4] = yt[5];
	for (i=0;i<=5;i++) yt[i]=y[i]+hh*dyt[i];
	step(f,yt,dym);
	dym[0] = yt[1];
	dym[2] = yt[3];
	dym[4] = yt[5];
	for (i=0;i<=5;i++) {
		yt[i]=y[i]+TAU*dym[i];
		dym[i] += dyt[i];
	}
	step(f,yt,dyt);
	dyt[0] = yt[1];
	dyt[2] = yt[3];
	dyt[4] = yt[5];
	for (i=0;i<=5;i++)
		yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
}

vector<double> performAction(double output, vector<double> state, int stepnum) { 
  
  vector<double>  dydx(6);
     
  /*--- Apply action to the simulated cart-pole ---*/
  for(int k=0; k<stepnum; k++) {
        for(int i=0; i<2; ++i){
            dydx[0] = state[1];
            dydx[2] = state[3];
            dydx[4] = state[5];
            step(output, state, dydx);
            rk4(output, state, dydx, state);
        }
  }
    
    return state;
  }
