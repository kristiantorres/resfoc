NPPATH=${CONDA_PREFIX}/lib/python3.7/site-packages/numpy/core

%.o: %.cpp
	g++ -std=c++11 -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I${CONDA_PREFIX}/include/python3.7m \
		-I/usr/local/Intel_16/tbb/include -I/home/joseph29/opt/FFTW/include \
		-I${NPPATH}/include -Wl,-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lpython3.7m \
		-I. -c -fPIC -o $@ $<

evntcre8: evntcre8.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g evntcre8.o evntcre8py.cpp -o ../evntcre8`${CONDA_PREFIX}/bin/python3-config --extension-suffix` \
		-L/home/joseph29/opt/FFTW/lib -L/usr/local/Intel_16/tbb/lib/intel64/gcc4.4 -ltbb -lfftw3f -lfftw3f_threads
