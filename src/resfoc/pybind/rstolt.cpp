
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "stretch.h"
#include "rstolt.h"
#include <cstring>

rstolt::rstolt(int nz, int nm, int nh, int nro, float dz, float dm, float dh, float dro, float oro) {
  /* Sizes */
  _nz = nz; _nm = nm; _nh = nh; _nro = 2*nro-1;
  /* Samplings */
  _dz = M_PI*dz; _dm = M_PI*dm; _dh = M_PI*dh; _dro = dro;
  /* Origin */
  _oro = oro - (nro-1)*dro;
}

void rstolt::resmig(float *dat, float *img, int nthrd) {

  /* Initialize stretch */
  stretch intrp = stretch(_nz,0.0,_dz,_nz,0.01);

  omp_set_num_threads(nthrd);
  /* Loop over rho */
#pragma omp parallel for default(shared)
  for(int iro = 0; iro < _nro; ++iro) {
    /* Temporary arrays */
    float *str = new float[_nz](); float *trc = new float[_nz]();
    float *mig = new float[_nz]();
    /* Compute rho */
    float vov = _oro + iro*_dro;
    /* Loop over sub-surface offset */
    for(int ih = 0; ih < _nh; ++ih) {
      float kh = ih*_dh;
      /* Loop over midpoint (image point) */
      for(int im = 0; im < _nm; ++im) {
        float km = im*_dm;
        /* Create the mapping z -> z' (loop from iz==2 to avoid kz=0) */
        for(int iz = 1; iz < _nz; ++iz) {
          float kz = iz*_dz;
          /* Dispersion relation */
          float kzh = kz*kz + kh*kh;
          float kzm = kz*kz + km*km;
          float zzs = (vov*vov) * (kzh*kzm) - (kz*kz) * ( (km-kh)*(km-kh) );
          float zzg = (vov*vov) * (kzh*kzm) - (kz*kz) * ( (km+kh)*(km+kh) );
          if(zzs > 0 && zzg > 0) {
            str[iz] = 0.5/kz * ( sqrt(zzs) + sqrt(zzg) );
          } else { /* Evanescent */
            str[iz] = -2.0*_dz;
          }
        }
        /* Do the migration for the mapping */
        memcpy(trc,&dat[ih*_nz*_nm + im*_nz],sizeof(float)*_nz);
        /* Initialize output trace to 0 */
        memset(mig, 0, sizeof(float)*_nz);
        intrp.apply(str, trc, mig);
        /* Copy to the output image volume */
        memcpy(&img[iro*_nz*_nm*_nh + ih*_nz*_nm + im*_nz],mig,sizeof(float)*_nz);
      }
    }
    /* Free memory */
    delete[] str; delete[] trc; delete[] mig;
  }

}
