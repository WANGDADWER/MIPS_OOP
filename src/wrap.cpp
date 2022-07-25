#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "CEOs.h"
#include "MIPS.h"

namespace py = pybind11;

PYBIND11_MODULE(example, m)
{
    
    /*
    py::class_<MIPS>(m, "MIPS")
    	    .def(py::init<int, int, int, int, int, bool>())
    	    .def("read_X_from_file", &MIPS::read_X_from_file)
            .def("read_Q_from_file", &MIPS::read_Q_from_file)
            .def("get_X", &MIPS::get_X)
            .def("get_Q", &MIPS::get_Q)
            .def("read_X_from_np",&MIPS::read_X_from_np)
            .def("read_Q_from_np",&MIPS::read_Q_from_np);
    */      
    py::class_<OneCEOs>(m, "OneCEOs")
            .def(py::init<int, int, int, int, int, bool, int>())
            .def("build_Index", &OneCEOs::build_Index)
            .def("find_TopK", &OneCEOs::find_TopK,py::return_value_policy::reference_internal)
            .def("read_X_from_file", &OneCEOs::read_X_from_file)
            .def("read_Q_from_file", &OneCEOs::read_Q_from_file)
            .def("get_X", &OneCEOs::get_X,py::return_value_policy::reference_internal)
            .def("get_Q", &OneCEOs::get_Q,py::return_value_policy::reference_internal)
            .def("read_X_from_np",&OneCEOs::read_X_from_np)
            .def("read_Q_from_np",&OneCEOs::read_Q_from_np);
         
    py::class_<TwoCEOs,OneCEOs>(m, "TwoCEOs")
            .def(py::init<int, int, int, int, int, bool, int>())
            .def("build_Index", &TwoCEOs::build_Index)
            .def("find_TopK", &TwoCEOs::find_TopK,py::return_value_policy::reference_internal);
            
    py::class_<sCEOsEst,OneCEOs>(m, "sCEOsEst")
            .def(py::init<int, int, int, int, int, bool, int, int>())
            .def("build_Index", &sCEOsEst::build_Index)
            .def("find_TopK", &sCEOsEst::find_TopK,py::return_value_policy::reference_internal);
            
    py::class_<sCEOsTA,sCEOsEst>(m, "sCEOsTA")
            .def(py::init<int, int, int, int, int, bool, int,int>())
            .def("build_Index", &sCEOsTA::build_Index)
            .def("find_TopK", &sCEOsTA::find_TopK,py::return_value_policy::reference_internal);
            
    py::class_<coCEOs,sCEOsEst>(m, "coCEOs")
            .def(py::init<int, int, int, int, int, bool, int,int,int>())
            .def("build_Index", &coCEOs::build_Index)
            .def("find_TopK", &coCEOs::find_TopK,py::return_value_policy::reference_internal);
}
