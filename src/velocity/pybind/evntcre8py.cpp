/**
 * Python interface to C++ geologic event creator
 * @author: Joseph Jennings
 * @version: 2020.01.22
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "evntcre8.h"

namespace py = pybind11;

PYBIND11_MODULE(evntcre8,m) {
  m.doc() = "Geologic event creation";

  py::class_<evntcre8>(m,"evntcre8")
      .def(py::init<int,int,float,float,float>(),
           py::arg("nx"),py::arg("ny"),
           py::arg("dx"), py::arg("dy"), py::arg("dz"))
      .def("expand",[](evntcre8 &ec8,
             int itop,
             int ibot,
             int nzin,
             py::array_t<int, py::array::c_style> lyrin,
             py::array_t<float, py::array::c_style> velin,
             int nzot,
             py::array_t<int, py::array::c_style> lyrot,
             py::array_t<float, py::array::c_style> velot
             )
             {
               ec8.expand(itop, ibot, nzin, lyrin.mutable_data(), velin.mutable_data(),
                   nzot, lyrot.mutable_data(), velot.mutable_data());
             },
             py::arg("itop"), py::arg("ibot"),
             py::arg("nzin"), py::arg("lyrin"), py::arg("velin"),
             py::arg("nzot"), py::arg("lyrot"), py::arg("velot")
          )
      .def("deposit",[](evntcre8 &ec8,
             float vel,
             float band1,
             float band2,
             float band3,
             float var,
             float layer,
             float layer_rand,
             float dev_layer,
             float dev_pos,
             int nzot,
             py::array_t<int,   py::array::c_style> lyrot,
             py::array_t<float, py::array::c_style> velot
             )
             {
               ec8.deposit(vel,
                   band1, band2, band3,
                   var, layer, layer_rand, dev_layer, dev_pos,
                   nzot, lyrot.mutable_data(), velot.mutable_data());
             },
             py::arg("vel"),
             py::arg("band1"), py::arg("band2"), py::arg("band3"),
             py::arg("var"), py::arg("layer"), py::arg("layer_rand"), py::arg("dev_layer"),
             py::arg("dev_pos"), py::arg("nzot"), py::arg("lyrot"), py::arg("velot")
      );
}
