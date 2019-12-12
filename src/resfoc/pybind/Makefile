PYPATH=/opt/anaconda/anaconda3/envs/py35
NPPATH=${PYPATH}/lib/python3.5/site-packages/numpy/core

%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

rstolt: rstolt.o stretch.o 
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g stretch.o rstolt.o rstoltpy.cpp -o ../rstolt`${PYPATH}/bin/python3-config --extension-suffix`
