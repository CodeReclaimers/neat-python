%module dpole_integrate
%{
#include "dpole_integrate.hpp"
%}

%include "std_vector.i"
// Instantiate templates used by dpole_integrate
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
}

// Include the header file with above prototypes
%include "dpole_integrate.hpp"
