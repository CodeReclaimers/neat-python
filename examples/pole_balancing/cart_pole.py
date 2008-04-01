# Cart pole module
from neat import nn
from math import cos, sin
from random import randint

class CartPole(object):
    def __init__(self, network, markov):
        
        self.__network = network
        
        # there are two types of double balancing experiment: 
        # 1. markovian: velocity information is provided to the network input
        # 2. non-markovian: no velocity is provided
        self.__markov = markov
        
        self.__dydx = []
        self.__state = []
               
    def evaluate(self, num_steps):
        
        steps = 0
        
        if(self.__markov):            
            self.__initial_state()            
            while(steps < num_steps):            
                inputs = [self.__state[0]/4.80,
                          self.__state[1]/2.00,
                          self.__state[2]/0.52,
                          self.__state[3]/2.00,
                          self.__state[4]/0.52,
                          self.__state[5]/2.00]
                
                output = self.__network.sactivate(inputs)    
                self.__perform_action(output[0], steps)   
                
                # cart's position, first pole's angle, second pole's angle
                #print "%f \t %f \t %f \t %f" %(self.__state[0], self.__state[2], self.__state[4], output[0]) 
                
                if(self.__outside_bounds()):
                    # network failed to solve the task
                    break                          
                steps += 1
                
        else:
            # non-markovian
            pass
            
            
        return float(steps) # higher number of steps = better fitness
        
    def __step(self, net_output, state):
        
        GRAVITY = 9.8
        MASSCART = 1.0

        FORCE_MAG = 10.0
        
        LENGTH_1 = 0.5
        MASSPOLE_1 = 0.1

        LENGTH_2 = 0.05;
        MASSPOLE_2 = 0.01;
        
        MUP = 0.000002
        
        dydx = [0 for i in xrange(6)]
        
        force = (net_output - 0.5) * FORCE_MAG * 2
        costheta_1 = cos(state[2])
        sintheta_1 = sin(state[2])
        gsintheta_1 = GRAVITY * sintheta_1
        costheta_2 = cos(state[4])
        sintheta_2 = sin(state[4])
        gsintheta_2 = GRAVITY * sintheta_2
        
        ml_1 = LENGTH_1 * MASSPOLE_1
        ml_2 = LENGTH_2 * MASSPOLE_2
        
        temp_1 = MUP * state[3] / ml_1
        temp_2 = MUP * state[5] / ml_2
        
        fi_1 = (ml_1 * state[3] * state[3] * sintheta_1) + \
                (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1))
                
        fi_2 = (ml_2 * state[5] * state[5] * sintheta_2) + \
                (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2))
               
        mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1))
        mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2))
        
        
        dydx[0] = state[1]
        dydx[1] = (force + fi_1 + fi_2) / (mi_1 + mi_2 + MASSCART)
        dydx[2] = state[3]
        dydx[3] = -0.75 * (dydx[1] * costheta_1 + gsintheta_1 + temp_1) / LENGTH_1
        dydx[4] = state[5]            
        dydx[5] = -0.75 * (dydx[1] * costheta_2 + gsintheta_2 + temp_2) / LENGTH_2
                      
        return dydx
        
    def __rk4(self, net_output, state, dydx):
        
        TAU = 0.02

        hh = TAU*0.5
        h6 = TAU/6.0
        
        yt = [0 for i in xrange(6)]
        next_state = [0 for i in xrange(6)]
        
        for i in xrange(6): 
            yt[i] = state[i] + hh*dydx[i]
        
        dyt = self.__step(net_output, yt)
        
        #dyt[0] = yt[1]
        #dyt[2] = yt[3]
        #dyt[4] = yt[5]
        
        for i in xrange(6):
            yt[i] = state[i] + hh*dyt[i]
            
        dym = self.__step(net_output, yt)
        
        #dym[0] = yt[1]
        #dym[2] = yt[3]
        #dym[4] = yt[5]
        
        for i in xrange(6):
            yt[i] = state[i] + TAU*dym[i]
            dym[i] += dyt[i]
            
        dyt = self.__step(net_output,yt)
        
        #dyt[0] = yt[1]
        #dyt[2] = yt[3]
        #dyt[4] = yt[5]
        
        for i in xrange(6):
            next_state[i] = state[i] + h6*(dydx[i] + dyt[i] + 2.0*dym[i])
            
        return next_state
        
    def __outside_bounds(self):

        failureAngle = 0.628329 #thirty_six_degrees

        #return self.__state[0] < -2.4 or self.__state[0] > 2.4 or \
        #       self.__state[2] < -failureAngle  or self.__state[2] > failureAngle or \
        #       self.__state[4] < -failureAngle  or self.__state[4] > failureAngle
        
        return  abs(self.__state[0]) > 2.4 or \
                abs(self.__state[2]) > failureAngle or \
                abs(self.__state[4]) > failureAngle
                

            
    def __initial_state(self):
          
        self.__dydx = [0, 0, 0, 0, 0, 0]

        # initial state (conditions)
        #self.__state = [0,
        #                0,
        #                0.07, # one_degree
        #                0,
        #                0,
        #                0]
                        
        if(True):
            r = randint
            self.__state = [(r(0, 2**31)%4800)/1000.0 - 2.4,
                            (r(0, 2**31)%2000)/1000.0 - 1,
                            (r(0, 2**31)%400)/1000.0 - 0.2,
                            (r(0, 2**31)%400)/1000.0 - 0.2,
                            (r(0, 2**31)%3000)/1000.0 - 1.5,
                            (r(0, 2**31)%3000)/1000.0 - 1.5]
        
        
    def __perform_action(self, net_output, stepnum):
                
        # Apply action to the simulated cart-pole
        for i in xrange(2):
            # next step   
            self.__dxdy  = self.__step(net_output, self.__state)        
            self.__state = self.__rk4(net_output, self.__state, self.__dydx)
