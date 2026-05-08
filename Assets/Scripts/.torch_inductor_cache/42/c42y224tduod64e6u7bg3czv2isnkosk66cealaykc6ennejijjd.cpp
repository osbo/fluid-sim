
#include "/orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       const int64_t* in_ptr1,
                       const int64_t* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(96L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
        }
        #pragma omp simd simdlen(8) 
        for(int64_t x0=static_cast<int64_t>(96L); x0<static_cast<int64_t>(109L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
            out_ptr0[static_cast<int64_t>(x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(96L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
        }
        #pragma omp simd simdlen(8) 
        for(int64_t x0=static_cast<int64_t>(96L); x0<static_cast<int64_t>(109L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr1[static_cast<int64_t>(x0)];
            out_ptr1[static_cast<int64_t>(x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(96L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
        }
        #pragma omp simd simdlen(8) 
        for(int64_t x0=static_cast<int64_t>(96L); x0<static_cast<int64_t>(109L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr2[static_cast<int64_t>(x0)];
            out_ptr2[static_cast<int64_t>(x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3488L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr3 + static_cast<int64_t>(x0));
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3488L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr4 + static_cast<int64_t>(x0));
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3488L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr5 + static_cast<int64_t>(x0));
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3488L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr6 + static_cast<int64_t>(x0));
        }
    }
}

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>

#ifndef _MSC_VER
#if __cplusplus < 202002L
// C++20 (earlier) code
// https://en.cppreference.com/w/cpp/language/attributes/likely
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer<T>::value, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1 && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(unlikely(!PyTuple_CheckExact(args)))
            throw std::runtime_error("tuple args required");
        if(unlikely(PyTuple_GET_SIZE(args) != 14))
            throw std::runtime_error("requires 14 args");
        kernel(parse_arg<int64_t*>(args, 0), parse_arg<int64_t*>(args, 1), parse_arg<int64_t*>(args, 2), parse_arg<float*>(args, 3), parse_arg<float*>(args, 4), parse_arg<float*>(args, 5), parse_arg<float*>(args, 6), parse_arg<int64_t*>(args, 7), parse_arg<int64_t*>(args, 8), parse_arg<int64_t*>(args, 9), parse_arg<float*>(args, 10), parse_arg<float*>(args, 11), parse_arg<float*>(args, 12), parse_arg<float*>(args, 13));Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }
    std::istringstream iss(str_addr);
    uintptr_t addr = 0;
    iss >> addr;
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    return PyModule_Create(&py_module);
}
