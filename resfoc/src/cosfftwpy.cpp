/**
 * Python interface to cosfftw
 * @author: Joseph Jennings
 * @version: 2020.11.08
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cosfftw.h"

namespace py = pybind11;

PYBIND11_MODULE(cosfftw,m) {
  m.doc() = "Cosine transform using FFTW";

  py::class_<cosfftw>(m,"cosfftw")
      .def(py::init([](int ndim, py::array_t<int, py::array::c_style> ns)
          { return new cosfftw(ndim, ns.mutable_data()); }))
      .def("fwd",[](cosfftw &cft,
          py::array_t<std::complex<float>, py::array::c_style> inp,
          py::array_t<std::complex<float>, py::array::c_style> out
          )
          {
            cft.fwd(inp.mutable_data(), out.mutable_data());
          },
          py::arg("inp"), py::arg("out")
      )
      .def("inv",[](cosfftw &cft,
          py::array_t<std::complex<float>, py::array::c_style> inp,
          py::array_t<std::complex<float>, py::array::c_style> out
          )
          {
            cft.fwd(inp.mutable_data(), out.mutable_data());
          },
          py::arg("inp"), py::arg("out")
      );
}
