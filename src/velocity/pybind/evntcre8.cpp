
#include <math.h>
#include <algorithm>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>
#include <fftw3.h>
#include "evntcre8.h"
#include "/opt/matplotlib-cpp/matplotlibcpp.h"

namespace plt = matplotlibcpp;

evntcre8::evntcre8(int nx, int ny, float dx, float dy, float dz) {
  _n2 = nx; _n3 = ny;
  _d2 = dx; _d3 = dy; _d1 = dz;
}

void evntcre8::expand(int itop, int ibot, int nzin, int *lyrin, float *velin, int nzot, int *lyrot, float *velot) {

  //TODO: should check nzot with nzin and itop and ibot
  /* Assign the expanded model indices in lyrot */
  tbb::parallel_for(tbb::blocked_range<int>(0, _n3),
      [&](const tbb::blocked_range<int>& r) {
    for (int i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < _n2; ++i2) {
        for (int i1 = 0; i1 < itop; ++i1) {
          lyrot[i3*nzot*_n2 + i2*nzot + i1] = -1;
          velot[i3*nzot*_n2 + i2*nzot + i1] = -1.0;
        }
        for (int i1 = 0; i1 < nzin; ++i1) {
          lyrot[i3*nzot*_n2 + i2*nzot + i1 + itop] = lyrin[i3*nzin*_n2 + i2*nzin + i1];
          velot[i3*nzot*_n2 + i2*nzot + i1 + itop] = velin[i3*nzin*_n2 + i2*nzin + i1];
        }
        for (int i1 = 0; i1 < ibot; ++i1) {
          lyrot[i3*nzot*_n2 + i2*nzot + i1 + nzot + itop] = -1;
          velot[i3*nzot*_n2 + i2*nzot + i1 + nzot + itop] = -1.0;
        }
      }
    }
  });

}

//TODO:  probably need to pass in the current event number
void evntcre8::deposit(float vel,
    float band1, float band2, float band3,
    float var, float layerT, float layer_rand, float dev_layer, float dev_pos,
    int nzot, int *lyrot, float *velot) {

  /* Find the thickness of the deposit based on the output layer and layer model*/
  int n1use = std::max(16, find_max_deposit(nzot,_n2,_n3,lyrot));

  /* Build a 1D function for defining the fine layering */
  int iold = 0;
  float vu = (1. + ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - .5) * dev_layer);
  std::vector<float> layv(nzot, 0.);

  float luse = layerT * (1 + (.5 * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))) * layer_rand);

  for (int i1 = 0; i1 < nzot; i1++) {
    int ii = (i1 - iold) * + _d1 / luse;
    if (ii != iold) {
      vu = (1. + ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - .5) * dev_layer);
    }
    iold = ii;
    luse = layerT * (1 + (.5 * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))) * layer_rand);
    layv[i1] = vu;
  }

  /* Start by filling with bandpassed random numbers */
  float *vtmp = new float[n1use*_n2*_n3]();
  fill_random(n1use, _n2, _n3, vtmp);
  bandpass(band1, band2, band3, n1use, _n2, _n3, vtmp);
  norm(n1use,_n2,_n3,vtmp,1);

  /* First compute the entire temporary array based on the max computed height */
  tbb::parallel_for(tbb::blocked_range<size_t>(0, _n3),
      [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < n1use; i1++) {
          vtmp[i3*_n2*n1use + i2*n1use + i1] = vel * layv[i1] * (1 + vtmp[i3*_n2*n1use + i2*n1use + i1] * dev_pos);
        }
      }
    }
  });

  /* Now fill in the layer based on the actual layer position (layer field) */
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, _n3),
      [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < _n2; ++i2) {
        int i1 = 0;
        while (i1 < nzot - 1 && lyrot[i3*_n2*nzot + i2*nzot + i1] == -1) i1++;
        for (int ia = 0; ia < i1; ++ia) {
          velot[i3*_n2*nzot + i2*nzot + ia] = vtmp[i3*_n2*n1use + i2*n1use + ia];
          lyrot[i3*_n2*nzot + i2*nzot + ia] = 1;
        }
      }
    }
  });

  /* Clean up */
  delete[] vtmp;
}

int evntcre8::find_max_deposit(int n1, int n2, int n3, int *lyrin) {
  int mx = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n3), int(0),
      [&](tbb::blocked_range<size_t> r, float value) -> float {
    for (size_t i3 = r.begin(); i3 != r.end(); i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        int i1 = 0;
        while (i1 < n1 - 1 && lyrin[i3*n1*n2 + i2*n1 + i1] == -1) i1++;
        if (value < i1) value = i1;
      }
    }
    return value;
  },
  [](int a, int b) {
    if (a > b) return a;
    return b;
  });
  return mx + 1;
}

void evntcre8::fill_random(int n1, int n2, int n3, float *velin) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n3),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i1 = 0; i1 < n1; ++i1) {
          velin[i3*n1*n2 + i2*n1 + i1] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 0.5;
        }
      }
    }
  });
}

void evntcre8::bandpass(float f1, float f2, float f3, int n1, int n2, int n3, float *velin) {

  std::vector<int> n(3, n1), nc(3, 1);

  n[1] = n2;
  n[2] = n3;
  nc[0] = n[0] / 2 + 1;
  nc[1] = n[1];
  nc[2] = n[2];
  size_t n123 = nc[0] * nc[1] * nc[2];

  fftwf_complex* tmp = new fftwf_complex[n123];
  fftwf_init_threads();
  std::vector<float> sc1 = find_lim1(nc[0], f1);
  std::vector<float> sc2 = find_lim2(nc[1], f2);
  std::vector<float> sc3 = find_lim2(nc[2], f3);

  fftwf_plan_with_nthreads(16);
  fftwf_plan forP = fftwf_plan_dft_r2c_3d(n[2], n[1], n[0], velin, tmp, FFTW_ESTIMATE);
  fftwf_plan bckP = fftwf_plan_dft_c2r_3d(n[2], n[1], n[0], tmp, velin, FFTW_ESTIMATE);

  size_t n12 = nc[0] * nc[1];
  float iN123 = 1. / (float)(n[0] * n[1] * n[2]);
  fftwf_execute(forP);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, nc[2]),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i3 = r.begin(); i3 < r.end(); i3++) {
      for (int i2 = 0; i2 < nc[1]; i2++) {
        for (int i1 = 0; i1 < nc[0]; i1++) {
          float sc = sc1[i1] * sc2[i2] * sc3[i3];
          tmp[i1 + i2 * nc[0] + i3 * n12][1] *= sc * iN123;
          tmp[i1 + i2 * nc[0] + i3 * n12][0] *= sc * iN123;
          ;
        }
      }
    }
  });

  fftwf_execute(bckP);

  delete[] tmp;
  fftwf_destroy_plan(forP);
  fftwf_destroy_plan(bckP);
  fftwf_cleanup_threads();
}

std::vector<float> evntcre8::find_lim2(int n, float f) {
  int b = std::min(n / 2 - 4, (int)(f * n / 2));
  int e = n - b;
  float cs[4] = {.9239, .7071, .3827, .0};
  std::vector<float> sc(n, 0.);
  for (int i = 0; i < b; i++) sc[i] = 1;
  for (int i = 0; i < 4; i++) sc[i + b] = cs[i];
  for (int i = 0; i < 4; i++) sc[e - 3 + i] = cs[3 - i];
  for (int i = e; i < n; i++) sc[i] = 1.;
  return sc;
}

std::vector<float> evntcre8::find_lim1(int n, float f) {
  int b = std::min(n - 4, (int)(f * n));
  float cs[4] = {.9239, .7071, .3827, .0};
  std::vector<float> sc(n, 0.);
  for (int i = 0; i < b; i++) sc[i] = 1;
  for (int i = 0; i < 4; i++) sc[i + b] = cs[i];
  return sc;
}

float evntcre8::find_max(int n1, int n2, int n3, float *arr) {

  float mx = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n3), float(0.0),
      [&](tbb::blocked_range<size_t> r, float value) -> float {
    for (size_t i3 = r.begin(); i3 != r.end(); i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i1 = 0; i1 < n1; i1++) {
          if (arr[i3*n1*n2 + i2*n1 + i1] > value) value = arr[i3*n1*n2 + i2*n1 + i1];
        }
      }
    }
    return value;
  },
  [](float a, float b) {
    if (a > b) return a;
    return b;
  });
  return mx;

}

float evntcre8::find_min(int n1, int n2, int n3, float *arr) {

  float mn = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n3), float(0.0),
      [&](tbb::blocked_range<size_t> r, float value) -> float {
    for (size_t i3 = r.begin(); i3 != r.end(); i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i1 = 0; i1 < n1; i1++) {
          if (arr[i3*n1*n2 + i2*n1 + i1] < value) value = arr[i3*n1*n2 + i2*n1 + i1];
        }
      }
    }
    return value;
  },
  [](float a, float b) {
    if (a < b) return a;
    return b;
  });
  return mn;
}

float evntcre8::find_absmax(int n1, int n2, int n3, float *arr) {
  return std::max(fabsf(find_max(n1,n2,n3,arr)), fabsf(find_min(n1,n2,n3,arr)));
}

void evntcre8::scale(int n1, int n2, int n3, float *arr, float sc) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n3),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i1 = 0; i1 < n1; i1++) {
          arr[i3*n1*n2 + i2*n1 + i1] *= sc;
        }
      }
    }

  });
}

void evntcre8::norm(int n1, int n2, int n3, float *arr, float sc) {
  float mx = find_absmax(n1,n2,n3,arr);
  scale(n1,n2,n3,arr,sc/mx);
}
