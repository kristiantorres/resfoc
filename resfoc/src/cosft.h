/**
 * Performs the cosine fourier transform (real to real)
 * A port of cosft.c from Madagascar written by Sergey Fomel
 * @author: Joseph Jennings
 * @version: 2020.04.02
 */
#ifndef COSFT_H_
#define COSFT_H_

#include "kiss_fftr.h"

class cosft {
  public:
  cosft();
  cosft(int n1_in);
  void fwd(float *q, int o1, int d1);
  void inv(float *q, int o1, int d1);
  ~cosft() {
    free(_fwd); free(_inv);
    delete [] _pp; delete[] _p;
  }

  private:
  int _nt, _nw, _n1;
  float *_p;
  kiss_fft_cpx *_pp;
  kiss_fftr_cfg _fwd, _inv;
};

#endif /* COSFT_H_ */
