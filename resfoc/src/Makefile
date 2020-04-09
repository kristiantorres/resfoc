%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -o $@ $<

%.o: %.cpp
	g++ -std=c++11 -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

depth2time: depth2time.o stretch.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g stretch.o depth2time.o depth2timepy.cpp -o ../depth2time`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

rstolt: rstolt.o stretch.o 
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g stretch.o rstolt.o rstoltpy.cpp -o ../rstolt`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

rstoltbig: rstoltbig.o cosft.o ficosft.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o kiss_fftr.o cosft.o ficosft.o rstoltbig.o rstoltbigpy.cpp \
		-o ../rstoltbig`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

ficosft: cosft.o ficosft.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o kiss_fftr.o cosft.o ficosft.o ficosftpy.cpp -o ../ficosft`${CONDA_PREFIX}/bin/python3-config --extension-suffix`
