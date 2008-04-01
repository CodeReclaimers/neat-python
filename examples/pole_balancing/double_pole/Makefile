# requires python-dev and swig

LIBTYPE = so
TARGETS = _dpole_integrate.$(LIBTYPE)

CC = g++ -o3
PYTHON=python2.5

LIBS=-L. -L/usr/include -l$(PYTHON)
#CFLAGS = -g -Wall
CFLAGS = -fPIC -shared

all: $(TARGETS)
	
_%.$(LIBTYPE): dpole_integrate_wrap.o
	$(CC) $(LIBS) -fPIC -shared $^ -o $@
	rm dpole_integrate_wrap.cxx 

%.o: %.cxx 
	g++ $(CFLAGS) -c dpole_integrate.hpp $< -I/usr/include/$(PYTHON)/

dpole_integrate_wrap.cxx: dpole_integrate.i 
	swig -c++ -python $<

clean:
	rm -f $(TARGETS) *_wrap.* *.py{c,o} *.so



