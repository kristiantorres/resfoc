#include "cosft_kiss.h"

cosft::cosft(int n1_in) {
  /* Get sizes */
  _n1 = n1_in;
  _nt = 2*kiss_fft_next_fast_size(_n1-1);
  _nw = _nt/2 + 1;
  /* Temporary arrays */
  _p = new float[_nt]();
  _pp = new kiss_fft_cpx[_nw]();
  /* Forward and inverse FFT objects */
  _fwd = kiss_fftr_alloc(_nt,0,NULL,NULL);
  _inv = kiss_fftr_alloc(_nt,1,NULL,NULL);
}

void cosft::fwd(float *q, int o1, int d1) {
  /* Copy data */
  for(int i = 0; i < _n1; ++i) {
    _p[i] = q[o1+i*d1];
  }
  /* Pad */
  for(int i = _n1; i < _nw; ++i) {
    _p[i] = 0.0;
  }
  for(int i = _nw; i < _nt; ++i) {
    _p[i] = _p[_nt-i];
  }
  /* Forward transform */
  kiss_fftr(_fwd,_p,_pp);
  /* Copy real part */
  for(int i = 0; i < _n1; ++i) {
    q[o1+i*d1] = _pp[i].r;
  }
}

void cosft::inv(float *q, int o1, int d1) {
  /* Copy data */
  for(int i = 0; i < _n1; ++i) {
    _pp[i].r = q[o1+i*d1];
    _pp[i].i = 0.0;
  }
  /* Pad */
  for(int i = _n1; i < _nw; ++i) {
    _pp[i].r = 0.0;
    _pp[i].i = 0.0;
  }
  /* Inverse transform */
  kiss_fftri(_inv,_pp,_p);
  /* Copy to output */
  for(int i = 0; i < _n1; ++i) {
    q[o1+i*d1] = _p[i]/_nt;
  }
}
