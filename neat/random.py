from random import Random

_inst = Random()
seed = _inst.seed
random = _inst.random
choice = _inst.choice
shuffle = _inst.shuffle
gauss = _inst.gauss
uniform = _inst.uniform
setstate = _inst.setstate
getstate = _inst.getstate