
#include <math.h>
#include <stdio.h>
#include "stretch.h"

stretch::stretch(int n1, float o1, float d1, int nd, float epsilon) {
  _nt = n1; _t0 = o1; _dt = d1;  //Input signal axis
  _nz = nd; // Output length
  _eps = epsilon;
}

void stretch::define(float *coord, bool *m, int *x, float *w, float *diag, float *offd) {

  /* Initialize elements of diagonal and off-diagonal */
  for(int it = 0; it < _nt; ++it) {
    diag[it] = 2.0*_eps; offd[it] = -_eps; // Regularization
  }

  //printf("t0=%g dt=%g\n",_t0,_dt);
  for(int id = 0; id < _nz; ++id) {
    /* Standard interpolation step */
    //printf("id=%d coord=%g\n",id,coord[id]);
    float rx = (coord[id] - _t0)/_dt;
    int ix = floor(rx);
    //printf("rx=%f ix=%d\n",rx,ix);
    rx -= ix;
    if(ix < 0 || ix > _nt -2) {
      //printf("rx=%g ix=%d\n",rx,ix);
      m[id] = true; continue;
    }
    x[id] = ix; m[id] = false; w[id] = rx;
    //printf("rx=%g ix=%d m=%d\n",rx,ix,m[id]);
    int i1 = ix; int i2 = i1 + 1;
    /* Interpolation weights */
    float w2 = rx; float w1 = 1 - w2;
    //printf("w1=%g w2=%g\n",w1,w2);
    /* Compute elements of matrices */
    diag[i1] += w1*w1;
    diag[i2] += w2*w2;
    offd[i1] += w1*w2;
    //printf("i1=%d i2=%d diag[i1]=%g diag[i2]=%g, offd[i1]=%g\n",i1,i2,diag[i1],diag[i2],offd[i1]);
    //printf("\n");
  }
}

void stretch::solve(bool *m, int *x, float *w, float *diag, float *offd, float *ord, float *mod) {

  /* Build right hand side */
  for(int id = 0; id < _nz; ++id) {
    if(m[id]) continue;
    int i1 = x[id]; int i2 = i1 + 1;
    float w2 = w[id]; float w1 = 1 - w2;
    mod[i1] += w1*ord[id];
    mod[i2] += w2*ord[id];
    //printf("id=%d x=%d\n",id,x[id]);
    //printf("w1=%g w2=%g\n",w1,w2);
    //printf("mi1=%g mi2=%g\n",mod[i1],mod[i2]);
    //printf("\n");
  }
  /* Invert system */
  tridiag(diag,offd,mod);
}

void stretch::apply(float *coord, float *ord, float *mod) {

  /* Allocate memory */
  bool *m     = new bool[_nz]();
  int  *x     = new int[_nz]();
  float *w    = new float[_nz]();
  float *diag = new float[_nt]();
  float *offd = new float[_nt]();

  /* Create linear system */
  define(coord,m,x,w,diag,offd);
  /* Solve linear system */
  solve(m,x,w,diag,offd,ord,mod);

  /* Free memory */
  delete[] x; delete[] m;
  delete[] w; delete[] diag; delete[] offd;

}

void stretch::tridiag(float *diag, float *offd, float *b) {

  float *d = new float[_nt]();
  float *o = new float[_nt]();

  d[0] = diag[0];
  for(int k = 1; k < _nt; ++k) {
    float t = offd[k-1]; o[k-1] = t/diag[k-1]; d[k] = diag[k] - t*o[k-1];
    //printf("k=%d t=%g o=%g d=%g\n",k,t,o[k-1],d[k]);
  }
  for(int k = 1; k < _nt; ++k) {
    b[k] = b[k] - o[k-1]*b[k-1];
  }
  b[_nt-1] = b[_nt-1]/d[_nt-1];
  for(int k = _nt-2; k >= 0; --k) {
    b[k] = b[k]/d[k] - o[k]*b[k+1];
  }

  delete[] d; delete[] o;

}
