#ifndef DISPLACEMENT_OP_CUDA_KERNEL_H_
#define DISPLACEMENT_OP_CUDA_KERNEL_H_

#include <math.h>
#include <stdint.h>
#include "stdio.h"

namespace bdm {

struct SimParams {
  uint32_t num_objects;
  int32_t grid_dimensions[3];
  uint32_t num_boxes_axis[3];
  uint32_t box_length;
  double timestep;
  double squared_radius;
  double max_displacement;
};

class DisplacementOpCudaKernel {
 public:
  DisplacementOpCudaKernel(uint32_t num_objects, uint32_t num_boxes);
  virtual ~DisplacementOpCudaKernel();

  void LaunchDisplacementKernel(
      double* positions, double* diameters, double* tractor_force,
      double* adherence, uint32_t* box_id, double* mass,
      uint32_t* starts, uint16_t* lengths, uint32_t* successors,
      double* cell_movements, SimParams host_params);

  void ResizeCellBuffers(uint32_t num_cells);
  void ResizeGridBuffers(uint32_t num_boxes);

 private:
  double* d_positions_ = NULL;
  double* d_diameters_ = NULL;
  double* d_mass_ = NULL;
  double* d_cell_movements_ = NULL;
  double* d_tractor_force_ = NULL;
  double* d_adherence_ = NULL;
  uint32_t* d_box_id_ = NULL;
  uint32_t* d_starts_ = NULL;
  uint16_t* d_lengths_ = NULL;
  uint32_t* d_successors_ = NULL;
};

}  // namespace bdm

#endif  // DISPLACEMENT_OP_CUDA_KERNEL_H_
