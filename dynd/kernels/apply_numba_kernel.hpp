#pragma once

#include <iostream>
#include "config.hpp"
#include <dynd/kernels/base_kernel.hpp>

namespace pydynd {
namespace nd {
  namespace functional {

    struct apply_numba_kernel
        : base_kernel<apply_numba_kernel, kernel_request_host, -1> {
      typedef apply_numba_kernel self_type;

      // Reference to the python int pointer
      PyObject *m_pyptr;
      // The concrete prototype the ckernel is for
      ndt::type m_proto;
      // The arrmeta
      const char *m_dst_arrmeta;
      std::vector<const char *> m_src_arrmeta;
      eval::eval_context m_ectx;

      apply_numba_kernel() : m_pyptr(NULL) {}

      ~apply_numba_kernel()
      {
        if (m_pyptr != NULL) {
          PyGILState_RAII pgs;
          Py_DECREF(m_pyptr);
        }
      }

      void single(char *dst, char *const *src)
      {
        // Numba ABI: ret ptr, excinfo ptr, env ptr, arg 0, arg 1, ...
        typedef int binfunc(int*, void*, void*, int, int);
        binfunc* f = (binfunc*)PyInt_AsLong(m_pyptr);
        int src_0 = *((int*)src[0]);
        int src_1 = *((int*)src[1]);
        (*f)((int*)dst, nullptr, nullptr, src_0, src_1);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        std::cout << "Not supported" << std::endl;
      }

      static intptr_t
      instantiate(const arrfunc_type_data *af_self, const ndt::arrfunc_type *af_tp,
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<nd::string, ndt::type> &tp_vars)
      {
        PyGILState_RAII pgs;

        self_type *self = self_type::make(ckb, kernreq, ckb_offset);
        self->m_proto = ndt::make_arrfunc(nsrc, src_tp, dst_tp);
        self->m_pyptr = *af_self->get_data_as<PyObject *>();
        Py_XINCREF(self->m_pyptr);
        self->m_dst_arrmeta = dst_arrmeta;
        self->m_src_arrmeta.resize(nsrc);
        copy(src_arrmeta, src_arrmeta + nsrc, self->m_src_arrmeta.begin());
        self->m_ectx = *ectx;
        return ckb_offset;
      }

      static void free(arrfunc_type_data *self_af)
      {
        PyObject *pyptr = *self_af->get_data_as<PyObject *>();
        if (pyptr) {
          PyGILState_RAII pgs;
          Py_DECREF(pyptr);
        }
      }
    };
  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd
