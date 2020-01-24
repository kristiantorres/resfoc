PYPATH=/opt/anaconda3/envs/py35
NPPATH=${PYPATH}/lib/python3.5/site-packages/numpy/core
MPL=/opt/matplotlib-cpp

%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I${PYPATH}/include/python3.5m \
		-I${NPPATH}/include -I${MPL}/ -Wl,-rpath=${PYTPATH}/lib -L${PYPATH}/lib -lpython3.5m \
		-I. -c -fPIC -o $@ $<

evntcre8: evntcre8.o 
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g evntcre8.o evntcre8py.cpp -o ../evntcre8`${PYPATH}/bin/python3-config --extension-suffix` \
		-ltbb -lfftw3f -lfftw3f_threads
