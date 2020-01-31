/**
 * Creates geological events as part of a geological
 * model building code
 * @author: Joseph Jennings
 * @version: 2020.01.22
 */
#ifndef EVNTCRE8_H_
#define EVNTCRE8_H_

class evntcre8 {
  public:
    evntcre8(int nx, int ny, float dx, float dy, float dz);
    void expand(int itop, int ibot, int nzin, int *lyrin, float *velin, int nzot, int *lyrot, float *velot);
    void deposit(float vel,
        float band1, float band2, float band3,
        float layerT, float layer_rand, float dev_layer, float dev_pos,
        int nzot, int *lyrot, float *velot);
    void fault(int nz, int *lyrin, float *velin, float azim, float begx, float begy, float begz, float dz, float daz,
        float thetashift, float perpdie, float distdie, float thetadie, float dir,
        int *lyrot, float *velot, float *lblout);
    int find_max_deposit(int n1, int n2, int n3, int *lyrin);
    void fill_random(int n1, int n2, int n3, float *velot);
    void bandpass(float f1, float f2, float f3, int n1, int n2, int n3, float *velin);
    std::vector<float> find_lim1(int n, float f);
    std::vector<float> find_lim2(int n, float f);
    float find_max(int n1, int n2, int n3, float *arr);
    float find_min(int n1, int n2, int n3, float *arr);
    float find_absmax(int n1, int n2, int n3, float *arr);
    void scale(int n1, int n2, int n3, float *arr, float sc);
    void norm(int n1, int n2, int n3, float *arr, float sc);
    void zder(int nz, float *lblin, float *lblot);

  private:
    int _n2, _n3;
    float _d2, _d3, _d1;
};

#endif /* EVNTCRE8_H_ */
