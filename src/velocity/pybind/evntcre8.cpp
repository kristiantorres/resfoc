
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

void evntcre8::fault(int nz, int *lyrin, float *velin, float azim, float begx, float begy, float begz, float dz, float daz,
    float thetashift, float perpdie, float distdie, float thetadie, float dir,
    int *lyrot, float *velot, float *lblot) {

  float dst1 = nz*_d1;
  float dst2 = _n2*_d2;
  float dst3 = _n3*_d3;

  /* Location of the beginning of the tear */
  float zbeg = dst1 * begz;
  float ybeg = dst3 * begy;
  float xbeg = dst2 * begx;

  // Die off distance from the tear perpendicular to the tear and radius direction
  distdie *= dst3;
  perpdie *= dst3;

  // Definition of pi
  float pi = atan(1.) * 4;

  float theta0 = atan2f(dz, daz);
  theta0 = theta0 + 2 * pi;
  theta0 = theta0 * 180. / pi;

  /* Azimuth normal to fault */
  if (azim < 0.) azim += 360.;  // Put azimuth in the 0-360
  azim *= pi / 180.;            // convert to radians
  float naz1   =  cosf(azim);   // Rotation matrix 1
  float naz2   = -sinf(azim);   // Rotation matrix 2
  float nperp1 =  sinf(azim);   // Rotation matrix 3
  float nperp2 =  cosf(azim);   // Rotation matrix 4

  /* Cylindrical coordinate 0,0,0 is at */
  float zcenter = zbeg - dz;
  float xcenter = xbeg - daz * cosf(azim);
  float ycenter = ybeg - daz * sinf(azim);

  float fullRadius = sqrtf(dz * dz + daz * daz);

  /* Create shift matrices */
  float *shiftx = new float[nz*_n2*_n3]();
  float *shifty = new float[nz*_n2*_n3]();
  float *shiftz = new float[nz*_n2*_n3]();

  /* Compute shifts */
  for (int i3 = 0; i3 < _n3; i3++) {
    float p3 = _d3 * i3 - ycenter;
    for (int i2 = 0; i2 < _n2; i2++) {
      float p2 = _d2 * i2 - xcenter;

      // Rotate coordinate x,y to along azimuth and perpendicular
      float azP = naz1 * p2 - naz2 * p3;
      float perpP = -nperp1 * p2 + nperp2 * p3;

      // Ratio die off along azimuth in 2D
      float ratioAz = fabsf(fullRadius - azP) / distdie;
      // Ratio die off along perp
      float ratioPerp = perpP / perpdie;

      float scalePerp = 1. - fabsf(ratioPerp);

      // If we are less than die off in perp and along azimuth
      for (int i1 = 0; i1 < nz; i1++) {
        shiftz[i3*_n2*nz + i2*nz + i1] = 0;
        shiftx[i3*_n2*nz + i2*nz + i1] = 0;
        shifty[i3*_n2*nz + i2*nz + i1] = 0;
      }

      if (fabsf(ratioPerp) < 1.) {
        for (int i1 = 0; i1 < nz; i1++) {
          float p1 = _d1 * i1 - zcenter;  // Distance away in z

          // Theta of our current point
          float thetaOld = atan2f(p1, azP) * 180. / pi;
          thetaOld += 360;
          float thetaCompare = atanf(p1 / azP) * 180. / pi + 360.;

          // True radius of current point
          float radius = sqrtf(azP * azP + p1 * p1);

          ratioAz = fabsf(fullRadius - radius) / distdie;
          float ratioTheta = fabsf(thetaCompare - theta0) / thetadie;

          if (ratioAz < 1. && ratioTheta < 1.) {
            float scaleAz = 1. - ratioAz;
            float scaleTheta = 1. - ratioTheta;

            // Shift in theta
            float shiftTheta = thetashift * scaleAz * scaleTheta * scalePerp;

            // New theta location
            float thetaNew = thetaOld + shiftTheta;
            if (dir < 0. || radius > fullRadius)
              thetaNew = thetaOld - shiftTheta;

            // Convert to polar coordinates
            float dPR = radius * cosf(thetaNew * pi / 180.);
            float newZ = radius * sinf(thetaNew * pi / 180.) + zcenter;

            // Now rotate back to standard coordinate system
            float newX = naz1 * dPR + naz2 * perpP + xcenter;
            float newY = nperp1 * dPR + nperp2 * perpP + ycenter;

            shiftz[i3*nz*_n2 + i2*nz + i1] = newZ - (_d1 * i1);
            lblot[i3*nz*_n2 + i2*nz + i1] = shiftz[i3*nz*_n2 + i2*nz + i1];
            shiftx[i3*nz*_n2 + i2*nz + i1] = newX - (_d2 * i2);
            //lblot[i3*nz*_n2 + i2*nz + i1] = shiftx[i3*nz*_n2 + i2*nz + i1];
            shifty[i3*nz*_n2 + i2*nz + i1] = newY - (_d3 * i3);
            //lblot[i3*nz*_n2 + i2*nz + i1] = shifty[i3*nz*_n2 + i2*nz + i1];
          }
        }
        bool found = false;
        int i1 = 0;
        while (i1 < nz - 1 && !found) {
          if (fabs(shiftz[i3*nz*_n2 + i2*nz + i1]) > 0.)
            found = true;
          else
            i1++;
        }
        if (found) {
          for (int i = 0; i <= i1; i++) {
            shiftz[i3*nz*_n2 + i2*nz + i] = shiftz[i3*nz*_n2 + i2*nz + i1];
            //lblot[i3*nz*_n2 + i2*nz + i]  = shiftz[i3*nz*_n2 + i2*nz + i1];
          }
        }
      }
    }
  }

  /* Compute output layer and velocity arrays */
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, _n3),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < nz; i1++) {
          int l1 = std::max(0, (int)(i1 - shiftz[i3*nz*_n2 + i2*nz + i1] / _d1 + .5));
          int l2 = std::max(0, (int)(i2 - shiftx[i3*nz*_n2 + i2*nz + i1] / _d2 + .5));
          int l3 = std::max(0, (int)(i3 - shifty[i3*nz*_n2 + i2*nz + i1] / _d3 + .5));
          if (l1 >=  nz) l1 =  nz - 1;
          if (l2 >= _n2) l2 = _n2 - 1;
          if (l3 >= _n3) l3 = _n3 - 1;
          if (l1 >= 0) {
            lyrot[i3*nz*_n2 + i2*nz + i1] = lyrin[l3*nz*_n2 + l2*nz + l1];
            velot[i3*nz*_n2 + i2*nz + i1] = velin[l3*nz*_n2 + l2*nz + l1];
          }
          else {
            lyrot[i3*nz*_n2 + i2*nz + i1] = -1;
            velot[i3*nz*_n2 + i2*nz + i1] =  0;
          }
        }
      }
    }
  });

  /* Free memory */
  delete[] shiftx; delete[] shifty; delete[] shiftz;

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

void evntcre8::zder(int nz, float *lblin, float *lblot) {

  for(int i3 = 0; i3 < _n3; ++i3) {
    for(int i2 = 0; i2 < _n2; ++i2) {
      for(int i1 = 0; i1 < nz; ++i1) {
        /* One-sided derivatives at the ends */
        if(i1 == 0) {
          lblot[i3*nz*_n2 + i2*nz + i1] = lblin[i3*nz*_n2 + i2*nz + i1+1] - lblin[i3*nz*_n2 + i2*nz + i1-0];
        } else if(i1 == nz-1) {
          lblot[i3*nz*_n2 + i2*nz + i1] = lblin[i3*nz*_n2 + i2*nz + i1+0] - lblin[i3*nz*_n2 + i2*nz + i1-1];
        } else {
          lblot[i3*nz*_n2 + i2*nz + i1] = 0.5*lblin[i3*nz*_n2 + i2*nz + i1+1] - 0.5*lblin[i3*nz*_n2 + i2*nz + i1-1];
        }
      }
    }
  }

}
