#include "cosfftw.h"
#include <cstring>

cosfftw::cosfftw(int ndim, int *ns) {
  _prod = nprod(ndim,ns);
  _inp = new std::complex<float>[_prod]();
  _out = new std::complex<float>[_prod]();

  /* Build the FFTW plans */
  _fplan = fftwf_plan_dft(ndim,ns,
                          reinterpret_cast<fftwf_complex*>(_inp),
                          reinterpret_cast<fftwf_complex*>(_out),
                          FFTW_FORWARD,FFTW_MEASURE);

  _iplan = fftwf_plan_dft(ndim,ns,
                          reinterpret_cast<fftwf_complex*>(_out),
                          reinterpret_cast<fftwf_complex*>(_inp),
                          FFTW_BACKWARD,FFTW_MEASURE);
}

void cosfftw::fwd(std::complex<float> *inp, std::complex<float> *out) {
  memcpy(_inp,inp,sizeof(std::complex<float>)*_prod);
  fftwf_execute(_fplan);
  memcpy(out,_out,sizeof(std::complex<float>)*_prod);
}

void cosfftw::inv(std::complex<float> *inp, std::complex<float> *out) {
  memcpy(_out,out,sizeof(std::complex<float>)*_prod);
  fftwf_execute(_iplan);
  memcpy(inp,_inp,sizeof(std::complex<float>)*_prod);
}

int cosfftw::nprod(int ndim, int *ns) {
  int nprod = 1;
  for(int idim = 0; idim < ndim; ++idim) {
    nprod *= ns[idim];
  }
  return nprod;
}
