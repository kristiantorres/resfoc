/**
 * Residual Stolt migration
 * @author: Joseph Jennings
 * @version: 2019.12.11
 */

#ifndef RSTOLT_H_
#define RSTOLT_H_

class rstolt {
public:
  rstolt(int nz, int nm, int nh, int nro, float dz, float dm, float dh, float dro, float oro);
  void resmig(float *dat, float *img, int nthrd);
  void convert2time(int nt, float ot, float dt, float *vel, float *depth, float *time);

private:
  int _nz, _nm, _nh, _nro;
  float _dz, _dm, _dh, _dro;
  float _oro;
};

#endif /* RSTOLT_H_ */
