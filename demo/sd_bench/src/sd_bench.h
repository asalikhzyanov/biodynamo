// -----------------------------------------------------------------------------
//
// Copyright (C) The BioDynaMo Project.
// All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// See the LICENSE file distributed with this work for details.
// See the NOTICE file distributed with this work for additional information
// regarding copyright ownership.
//
// -----------------------------------------------------------------------------

#ifndef DEMO_SD_BENCH_H_
#define DEMO_SD_BENCH_H_

#include <chrono>
#include "biodynamo.h"

namespace bdm {

// -----------------------------------------------------------------------------
// This model creates a grid of 128x128x128 cells. Each cell grows untill a
// specific volume, after which it proliferates (i.e. divides).
// -----------------------------------------------------------------------------

// 1. Define compile time parameter
template <typename Backend>
struct CompileTimeParam : public DefaultCompileTimeParam<Backend> {};

inline int Simulate(int argc, const char** argv) {
  // 2. Create new simulation
  Simulation<> simulation(argc, argv);

  // 3. Define initial model - in this example: 3D grid of cells
  size_t cells_per_dim = 128;
  auto construct = [](const std::array<double, 3>& position) {
    Cell cell(position);
    cell.SetDiameter(30);
    cell.SetAdherence(0.4);
    cell.SetMass(1.0);
    return cell;
  };
  ModelInitializer::Grid3D(cells_per_dim, 20, construct);

  // 4. Run simulation for one timestep.
  std::cout << "Number of simulation objects,Number of threads,Physics time" << std::endl;
  std::string threads;
  if (std::getenv("OMP_NUM_THREADS")) {
    threads = std::string(std::getenv("OMP_NUM_THREADS"));
  } else {
    std::cout << "You didn't set the number of threads with OMP_NUM_THREADS!" << std::endl;
    exit(1);
  }
  std::cout << cells_per_dim * cells_per_dim * cells_per_dim << "," << threads << ",";

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  simulation.GetScheduler()->Simulate(1);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
  return 0;
}

}  // namespace bdm

#endif  // DEMO_SD_BENCH_H_
