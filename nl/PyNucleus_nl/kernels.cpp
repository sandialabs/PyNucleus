/////////////////////////////////////////////////////////////////////////////////////
// Copyright 2021 National Technology & Engineering Solutions of Sandia,           //
// LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           //
// U.S. Government retains certain rights in this software.                        //
// If you want to use this code, please refer to the README.rst and LICENSE files. //
/////////////////////////////////////////////////////////////////////////////////////


#include "kernels.hpp"
#include "math.h"
#include <iostream>

kernel_t::kernel_t() {
  // pass
}

REAL_t kernel_t::eval(REAL_t *x, REAL_t *y) {
  std::cout << "Calling base\n";
  return 0.;
};

fractional_kernel_t::fractional_kernel_t(REAL_t s_,
                                         REAL_t C_) {
  s = s_;
  C = C_;
}

REAL_t fractional_kernel_t::eval(REAL_t *x, REAL_t *y){
  return C * pow(abs(*x-*y), -1.-2.*s);
}


callback_kernel_t::callback_kernel_t(kernel_callback_t kernel_callback_, void *user_data_) {
  kernel_callback = kernel_callback_;
  user_data = user_data_;
}

REAL_t callback_kernel_t::eval(REAL_t *x, REAL_t *y){
  return kernel_callback(x, y, user_data);
}

