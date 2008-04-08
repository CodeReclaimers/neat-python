from neat.nn import nn_pure as nn

try: 
    import psyco; psyco.full()
except ImportError:
    pass

class CTNeuron(nn.Neuron):
    """ Continuous-time neuron model based on:
    
        Beer, R. D. and Gallagher, J.C. (1992). 
        Evolving Dynamical Neural Networks for Adaptive Behavior. 
        Adaptive Behavior 1(1):91-122. 
    """
    def __init__(self, neurontype, id = None, bias = 0.0, response = 1.0, activation_type = 'exp', tau = 1.0):
        super(CTNeuron, self).__init__(neurontype, id, bias, response, activation_type)

        # decay rate
        self.__tau  = tau 
        # needs to set the initial state (initial condition for the ODE)
        self.__state = 0.1 #TODO: Verify what's the "best" initial state
        # fist output
        self._output = nn.sigmoid(self.__state + self._bias, self._response, self._activation_type)
        # integration step
        self.__dt = 0.05 # depending on the tau constant, the integration step must 
                         # be adjusted accordingly to avoid numerical instability
                         
    def set_integration_step(self, step):
        self.__dt = step

    def set_init_state(self, state):
        self.__state = state
        self._output = nn.sigmoid(self.__state + self._bias, self._response, self._activation_type)

    def activate(self):
        " Updates neuron's state for a single time-step. "
        if(len(self._synapses) > 0):            
            self.__update_state()
            return nn.sigmoid(self.__state + self._bias, self._response, self._activation_type)
        else:
            return self._output # in case it's a sensor

    def __update_state(self):
        """ Returns neuron's next state using Forward-Euler method.
            Future plans: integrate using RK4.
        """
        self.__state += self.__dt*(1.0/self.__tau)*(-self.__state + self._update_activation())
               

def create_phenotype(chromo):
    """ Receives a chromosome and returns its phenotype (a CTRNN) """ 
    neurons_list = [CTNeuron(ng._type, 
                             ng._id, 
                             ng._bias, 
                             ng._response, 
                             ng._activation_type, 
                             ng._time_constant) \
                    for ng in chromo._node_genes]
        
    conn_list = [(cg.innodeid, cg.outnodeid, cg.weight) \
                  for cg in chromo.conn_genes if cg.enabled] 
                  
    return nn.Network(neurons_list, conn_list)

if __name__ == "__main__":
    # This example follows from Beer's C++ source code available at: 
    # http://mypage.iu.edu/~rdbeer/
    from neat import config
    config.Config.nn_activation = 'exp'  # using exponential sigmoid
    config.Config.feedforward = 0        # allow for recurrent topologies

    # create two output neurons (they won't receive any external inputs)
    N1 = CTNeuron('OUTPUT', id = 1, bias = -2.75, response = 1.0, activation_type = 'exp', tau = 1.0)
    N2 = CTNeuron('OUTPUT', id = 2, bias = -1.75, response = 1.0, activation_type = 'exp', tau = 1.0)
    N1.set_init_state(-0.0840006)
    N2.set_init_state(-0.408035)    
    neurons_list = [N1, N2]    
    # create some synapses
    conn_list = [(1, 1, 4.5), (1, 2, -1.0), (2, 1, 1.0), (2, 2, 4.5)]    
    # create the network
    net = nn.Network(neurons_list, conn_list)
    # activates the network
    for i in xrange(10):
        print net.pactivate()
    
    
