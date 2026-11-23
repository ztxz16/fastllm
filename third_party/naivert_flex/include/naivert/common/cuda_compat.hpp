#pragma once

// Reverse evidence:
// - The real kernel sources are CUDA device code.
//
// Inference boundary:
// - The replica is also used by host-side protocol tests.  These compatibility
//   macros let the sketch headers compile with a normal C++ compiler when the
//   test does not include real CUDA entry files.

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#endif
