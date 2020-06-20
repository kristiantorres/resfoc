PYPATH=/opt/anaconda3/envs/py35
NPPATH=${PYPATH}/lib/python3.5/site-packages/numpy/core

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -o $@ $<

%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall -I${PYPATH}/include/python3.5m \
		-I${NPPATH}/include -Wl,-rpath=${PYTPATH}/lib -L${PYPATH}/lib -lpython3.5m -c $< -o $@

rstolt: rstolt.o stretch.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g stretch.o rstolt.o rstoltpy.cpp -o ../rstolt`${PYPATH}/bin/python3-config --extension-suffix`

rstoltbig: rstoltbig.o ficosft.o cosft.o kiss_fft.o kiss_fftr.o stretch.o 
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g stretch.o kiss_fft.o kiss_fftr.o cosft.o ficosft.o rstoltbig.o rstoltbigpy.cpp -o \
		../rstoltbig`${PYPATH}/bin/python3-config --extension-suffix`

rhoshifts: rhoshifts.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g rhoshifts.o rhoshiftspy.cpp -o ../rhoshifts`${PYPATH}/bin/python3-config --extension-suffix`

depth2time: depth2time.o stretch.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g stretch.o depth2time.o depth2timepy.cpp -o ../depth2time`${PYPATH}/bin/python3-config --extension-suffix`

ficosft: cosft.o ficosft.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o kiss_fftr.o cosft.o ficosft.o ficosftpy.cpp -o ../ficosft`${PYPATH}/bin/python3-config --extension-suffix`

pickscan: dynprog.o pickscan.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g dynprog.o pickscan.o pickscanpy.cpp -o ../pickscan`${PYPATH}/bin/python3-config --extension-suffix`

