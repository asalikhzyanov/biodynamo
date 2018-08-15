// #include "samples/common/inc/helper_math.h"
// #include "gpu/displacement_op_cuda_kernel.h"

// #include "assert.h"

// #define GpuErrchk(ans) { GpuAssert((ans), __FILE__, __LINE__); }
// inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
//    if (code != cudaSuccess)
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (code == cudaErrorInsufficientDriver) {
//         printf("This probably means that no CUDA-compatible GPU has been detected. Consider setting the use_opencl flag to \"true\" in the bmd.toml file to use OpenCL instead.\n");
//       }
//       if (abort) exit(code);
//    }
// }

// __device__ int3 GetBoxCoordinates2(double3 pos, int32_t* grid_dimensions, uint32_t box_length) {
//   int3 box_coords;
//   box_coords.x = (floor(pos.x) - grid_dimensions[0]) / box_length;
//   box_coords.y = (floor(pos.y) - grid_dimensions[1]) / box_length;
//   box_coords.z = (floor(pos.z) - grid_dimensions[2]) / box_length;
//   return box_coords;
// }

// __device__ int3 GetBoxCoordinates(uint32_t box_idx, uint32_t* num_boxes_axis) {
//   int3 box_coord;
//   box_coord.z = box_idx / (num_boxes_axis[0]*num_boxes_axis[1]);
//   uint32_t remainder = box_idx % (num_boxes_axis[0]*num_boxes_axis[1]);
//   box_coord.y = remainder / num_boxes_axis[0];
//   box_coord.x = remainder % num_boxes_axis[0];
//   return box_coord;
// }

// __device__ uint32_t GetBoxId(int3 bc, uint32_t* num_boxes_axis) {
//   return bc.z * num_boxes_axis[0]*num_boxes_axis[1] + bc.y * num_boxes_axis[0] + bc.x;
// }

// __device__ void GetMooreBoxIds(uint32_t box_idx, uint32_t* ret, uint32_t* num_boxes_axis) {
//   const int3 moore_offset[27] = {
//     make_int3(-1, -1, -1), make_int3(0, -1, -1), make_int3(1, -1, -1),
//     make_int3(-1, 0, -1),  make_int3(0, 0, -1),  make_int3(1, 0, -1),
//     make_int3(-1, 1, -1),  make_int3(0, 1, -1),  make_int3(1, 1, -1),
//     make_int3(-1, -1, 0),  make_int3(0, -1, 0),  make_int3(1, -1, 0),
//     make_int3(-1, 0, 0),   make_int3(0, 0, 0),   make_int3(1, 0, 0),
//     make_int3(-1, 1, 0),   make_int3(0, 1, 0),   make_int3(1, 1, 0),
//     make_int3(-1, -1, 1),  make_int3(0, -1, 1),  make_int3(1, -1, 1),
//     make_int3(-1, 0, 1),   make_int3(0, 0, 1),   make_int3(1, 0, 1),
//     make_int3(-1, 1, 1),   make_int3(0, 1, 1),   make_int3(1, 1, 1)};

//   int3 box_coords = GetBoxCoordinates(box_idx, num_boxes_axis);
//   for (unsigned i = 0; i < 27; i++) {
//     ret[i] = GetBoxId(box_coords + moore_offset[i], num_boxes_axis);
//   }
// }

// __device__ void CheckAdherence(uint32_t cidx, double* result,
//               double timestep, double max_displacement, double* mass, double* adherence) {
//   // Mass needs to non-zero!
//   // double mh = timestep / mass[cidx];

//   // if (length(collision_force) > adherence[cidx]) {
//   //   result[3*cidx + 0] += collision_force.x * mh;
//   //   result[3*cidx + 1] += collision_force.y * mh;
//   //   result[3*cidx + 2] += collision_force.z * mh;

//   //   if (length(collision_force) * mh > max_displacement) {
//   //     result[3*cidx + 0] = max_displacement;
//   //     result[3*cidx + 1] = max_displacement;
//   //     result[3*cidx + 2] = max_displacement;
//   //   }
//   // }
// }

// __device__ void ComputeForce(uint32_t cidx, uint32_t nidx, double3* positions,
//       double* diameters, double3* force) {
//   double r1 = 0.5 * diameters[cidx];
//   double r2 = 0.5 * diameters[nidx];
//   // We take virtual bigger radii to have a distant interaction, to get a desired density.
//   const double additional_radius = 10.0 * 0.15;
//   r1 += additional_radius;
//   r2 += additional_radius;

//   double3 comp = positions[cidx] - positions[nidx];
//   double center_distance = length(comp);

//   // the overlap distance (how much one penetrates in the other)
//   double delta = r1 + r2 - center_distance;

//   // to avoid a division by 0 if the centers are (almost) at the same location
//   if (delta < 0) { return; }

//   if (center_distance < 0.00000001) {
//     *force = make_double3(42, 42, 42);
//   }

//   // the force itself
//   double r = (r1 * r2) / (r1 + r2);
//   double gamma = 1; // attraction coeff
//   double k = 2;     // repulsion coeff
//   double f = k * delta - gamma * sqrt(r * delta);

//   double module = f / center_distance;
//   *force = module * comp;
// }

// // Create arrays of cell data for each box
// // Create packages of these arrays for the 27 Moore boxes
// // Create dictionary <local_id, real_id> of each cell
// // kernel1<<<num_boxes, 1>>>(all_pos, all_diameter, box_starts, box_lengths, successors, *cache_location)
// __constant__ bdm::SimParams params;

// __global__ void collide(
//         double* positions,
//         double* diameters,
//         double* tractor_force,
//         double* adherence,
//         uint32_t* box_id,
//         double* mass,
//         uint32_t* starts,
//         uint16_t* lengths,
//         uint32_t* successors,
//         double* result) {
//   __shared__ double3 shared_pos[27*4];
//   __shared__ double3 shared_res[27*4];
//   __shared__ double  shared_dia[27*4];
//   __shared__ uint32_t moore_boxes[27];
//   __shared__ uint32_t i;
//   // unique block index inside a 3D block grid
//   const unsigned long long int blockId = blockIdx.x
//                                        + blockIdx.y * gridDim.x
//                                        + blockIdx.z * gridDim.x * gridDim.y;
//   if (blockIdx.x > 0 && blockIdx.x < gridDim.x && blockIdx.y > 0 && blockIdx.y < gridDim.y && blockIdx.z > 0 && blockIdx.z < gridDim.z) {
//     if (threadIdx.x == 0) {
//       i = 0;
//       uint32_t cidx = starts[blockId];
//       for (uint32_t nb = 0; nb < lengths[blockId]; nb++) {
//         shared_pos[i] = make_double3(positions[3*cidx], positions[3*cidx+1], positions[3*cidx+2]);
//         shared_dia[i] = diameters[cidx];
//         cidx = successors[cidx];
//         i++;
//       }
//       GetMooreBoxIds(blockId, &moore_boxes[0], params.num_boxes_axis);
//     }

//     if (threadIdx.x < 27 && (threadIdx.x != 12)) {
//       uint32_t bidx = moore_boxes[threadIdx.x];
//       if (bidx < gridDim.x * gridDim.y * gridDim.z) {
//         uint32_t cidx = starts[bidx];
//         for (uint32_t nb = 0; nb < lengths[bidx]; nb++) {  // for all cells per box
//           int li = atomicAdd(&i, 1);
//           shared_pos[li] = make_double3(positions[3 * cidx], positions[3 * cidx + 1], positions[3 * cidx + 2]);
//           shared_dia[li] = diameters[cidx];
//           cidx = successors[cidx];
//         }
//       }
//     }

//     __syncthreads();
//     // the local id of a thread within a block [0, blockDim.x - 1]
//     uint32_t nb_idx = threadIdx.x;

//     shared_res[nb_idx] = make_double3(0, 0, 0);

//     for (uint32_t cc = 0; cc < lengths[blockId]; cc++) {  // for all cells in middle box
//       if (nb_idx < i && cc != nb_idx) {
//         ComputeForce(cc, nb_idx, &shared_pos[0], &shared_dia[0], &shared_res[0]);
//         // CheckAdherence(cidx, result, timestep[0], max_displacement[0], mass, adherence);
//       }
//     }

//     if (threadIdx.x < i) {

//     }
//   }
// }

// bdm::DisplacementOpCudaKernel::DisplacementOpCudaKernel(uint32_t num_objects, uint32_t num_boxes) {
//   GpuErrchk(cudaMalloc(&d_positions_, 3 * num_objects * sizeof(double)));
//   GpuErrchk(cudaMalloc(&d_diameters_, num_objects * sizeof(double)));
//   GpuErrchk(cudaMalloc(&d_tractor_force_, 3 * num_objects * sizeof(double)));
//   GpuErrchk(cudaMalloc(&d_adherence_, num_objects * sizeof(double)));
//   GpuErrchk(cudaMalloc(&d_box_id_, num_objects * sizeof(uint32_t)));
//   GpuErrchk(cudaMalloc(&d_mass_, num_objects * sizeof(double)));
//   GpuErrchk(cudaMalloc(&d_starts_, num_boxes * sizeof(uint32_t)));
//   GpuErrchk(cudaMalloc(&d_lengths_, num_boxes * sizeof(uint16_t)));
//   GpuErrchk(cudaMalloc(&d_successors_, num_objects * sizeof(uint32_t)));
//   GpuErrchk(cudaMalloc(&d_cell_movements_, 3 * num_objects * sizeof(double)));
// }

// void bdm::DisplacementOpCudaKernel::LaunchDisplacementKernel(double* positions, double* diameters, double* tractor_force,
//                     double* adherence, uint32_t* box_id, double* mass,
//                     uint32_t* starts, uint16_t* lengths, uint32_t* successors,
//                     double* cell_movements, SimParams host_params) {
//   uint32_t num_boxes = host_params.num_boxes_axis[0] * host_params.num_boxes_axis[1] * host_params.num_boxes_axis[2];

//   GpuErrchk(cudaMemcpy(d_positions_, 		positions, 3 * host_params.num_objects * sizeof(double), cudaMemcpyHostToDevice));
//   GpuErrchk(cudaMemcpy(d_diameters_, 		diameters, host_params.num_objects * sizeof(double), cudaMemcpyHostToDevice));
//   GpuErrchk(cudaMemcpy(d_tractor_force_, 	tractor_force, 3 * host_params.num_objects * sizeof(double), cudaMemcpyHostToDevice));
//   GpuErrchk(cudaMemcpy(d_adherence_,     adherence, host_params.num_objects * sizeof(double), cudaMemcpyHostToDevice));
//   GpuErrchk(cudaMemcpy(d_box_id_, 		box_id, host_params.num_objects * sizeof(uint32_t), cudaMemcpyHostToDevice));
//   GpuErrchk(cudaMemcpy(d_mass_, 				mass, host_params.num_objects * sizeof(double), cudaMemcpyHostToDevice));
//   GpuErrchk(cudaMemcpy(d_starts_, 			starts, num_boxes * sizeof(uint32_t), cudaMemcpyHostToDevice));
//   GpuErrchk(cudaMemcpy(d_lengths_, 			lengths, num_boxes * sizeof(uint16_t), cudaMemcpyHostToDevice));
//   GpuErrchk(cudaMemcpy(d_successors_, 		successors, host_params.num_objects * sizeof(uint32_t), cudaMemcpyHostToDevice));

//   cudaMemcpyToSymbol(params, &host_params, sizeof(SimParams));
//   int blockSize;
//   dim3 gridSize;

//   gridSize.x = host_params.num_boxes_axis[0];
//   gridSize.y = host_params.num_boxes_axis[1];
//   gridSize.z = host_params.num_boxes_axis[2];

//   blockSize = 27*4;

//   // printf("gridSize = %d  |  blockSize = %d\n", gridSize, blockSize);
//   collide<<<gridSize, blockSize>>>(d_positions_, d_diameters_, d_tractor_force_,
//     d_adherence_, d_box_id_, d_mass_, d_starts_, d_lengths_, d_successors_,
//     d_cell_movements_);

//   // We need to wait for the kernel to finish before reading back the result
//   cudaDeviceSynchronize();
//   cudaMemcpy(cell_movements, d_cell_movements_, 3 * host_params.num_objects * sizeof(double), cudaMemcpyDeviceToHost);
// }

// void bdm::DisplacementOpCudaKernel::ResizeCellBuffers(uint32_t num_cells) {
//   cudaFree(d_positions_);
//   cudaFree(d_diameters_);
//   cudaFree(d_tractor_force_);
//   cudaFree(d_adherence_);
//   cudaFree(d_box_id_);
//   cudaFree(d_mass_);
//   cudaFree(d_successors_);
//   cudaFree(d_cell_movements_);

//   cudaMalloc(&d_positions_, 3 * num_cells * sizeof(double));
//   cudaMalloc(&d_diameters_, num_cells * sizeof(double));
//   cudaMalloc(&d_tractor_force_, 3 * num_cells * sizeof(double));
//   cudaMalloc(&d_adherence_, num_cells * sizeof(double));
//   cudaMalloc(&d_box_id_, num_cells * sizeof(uint32_t));
//   cudaMalloc(&d_mass_, num_cells * sizeof(double));
//   cudaMalloc(&d_successors_, num_cells * sizeof(uint32_t));
//   cudaMalloc(&d_cell_movements_, 3 * num_cells * sizeof(double));
// }

// void bdm::DisplacementOpCudaKernel::ResizeGridBuffers(uint32_t num_boxes) {
//   cudaFree(d_starts_);
//   cudaFree(d_lengths_);

//   cudaMalloc(&d_starts_, num_boxes * sizeof(uint32_t));
//   cudaMalloc(&d_lengths_, num_boxes * sizeof(uint16_t));
// }

// bdm::DisplacementOpCudaKernel::~DisplacementOpCudaKernel() {
//   cudaFree(d_positions_);
//   cudaFree(d_diameters_);
//   cudaFree(d_tractor_force_);
//   cudaFree(d_adherence_);
//   cudaFree(d_box_id_);
//   cudaFree(d_mass_);
//   cudaFree(d_starts_);
//   cudaFree(d_lengths_);
//   cudaFree(d_successors_);
//   cudaFree(d_cell_movements_);
// }
