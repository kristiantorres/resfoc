/**
 * Attempt to replace kiss cosft with fftw
 * @author: Joseph Jennings
 * @version: 2020.11.08
 */

#ifndef COSFFTW_H_
#define COSFFTW_H_

#include <complex>
#include <fftw3.h>

class cosfftw {
  public:
    cosfftw(int ndim, int* ns);
    void fwd(std::complex<float> *inp, std::complex<float> *out);
    void inv(std::complex<float> *inp, std::complex<float> *out);
    int nprod(int indim, int *ns);
   ~cosfftw(){
     delete[] _inp; delete[] _out;
     fftwf_destroy_plan(_fplan); fftwf_destroy_plan(_iplan);
   };
  private:
   int _prod;
   std::complex<float> *_inp, *_out;
   fftwf_plan _fplan, _iplan;
};

#endif /* RESFOC_SRC_COSFFTW_H_ */
