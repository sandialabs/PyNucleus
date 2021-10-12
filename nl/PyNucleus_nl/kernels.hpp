/////////////////////////////////////////////////////////////////////////////////////
// Copyright 2021 National Technology & Engineering Solutions of Sandia,           //
// LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           //
// U.S. Government retains certain rights in this software.                        //
// If you want to use this code, please refer to the README.rst and LICENSE files. //
/////////////////////////////////////////////////////////////////////////////////////


#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <stdint.h>
#include "myTypes.h"

typedef REAL_t (*kernel_callback_t)(REAL_t *x, REAL_t *y, void *user_data);

class kernel_t{
public:
  kernel_t();
  virtual REAL_t eval(REAL_t *x, REAL_t *y);
};


class fractional_kernel_t : public kernel_t {
public:
  fractional_kernel_t(REAL_t s_, REAL_t C_);
  REAL_t eval(REAL_t *x, REAL_t *y);
private:
  REAL_t s;
  REAL_t C;
};


class callback_kernel_t : public kernel_t {
public:
  callback_kernel_t(kernel_callback_t kernel_callback_, void* user_data_);
  REAL_t eval(REAL_t *x, REAL_t *y);
private:
  kernel_callback_t kernel_callback;
  void* user_data;
};


#endif
