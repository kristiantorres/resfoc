NPPATH=${CONDA_PREFIX}/lib/python3.7/site-packages/numpy/core

%.o: %.cpp
	g++ -std=c++11 -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I${CONDA_PREFIX}/include/python3.5m \
		-I${NPPATH}/include -Wl,-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lpython3.5m \
		-I. -c -fPIC -o $@ $<

evntcre8: evntcre8.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g evntcre8.o evntcre8py.cpp -o ../evntcre8`${CONDA_PREFIX}/bin/python3-config --extension-suffix` \
		-ltbb -lfftw3f -lfftw3f_threads
