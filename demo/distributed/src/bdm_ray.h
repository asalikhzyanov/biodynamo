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

#ifndef DEMO_DISTRIBUTED_BDM_RAY_H_
#define DEMO_DISTRIBUTED_BDM_RAY_H_

#include <local_scheduler/local_scheduler_client.h>
#include <plasma/client.h>

#include "backend.h"
#include "partitioner.h"

constexpr char kSimulationStartMarker[] = "aaaaaaaaaaaaaaaaaaaa";
constexpr char kSimulationEndMarker[] = "bbbbbbbbbbbbbbbbbbbb";

namespace bdm {

using ResourceManagerPtr = std::shared_ptr<ResourceManager<>>;
using SurfaceToVolume = std::pair<Surface, ResourceManagerPtr>;
// Not a map, but a constant size linear array.
using SurfaceToVolumeMap = std::array<SurfaceToVolume, 27>;

class RayScheduler : public Scheduler<Simulation<>> {
 public:
  using super = Scheduler<Simulation<>>;

  /// Runs one simulation timestep for `box` in `step` with global `bound`.
  void SimulateStep(long step, long box, bool last_iteration, const Box& bound);

  /// Initiates a distributed simulation and waits for its completion.
  ///
  /// This method will:
  ///
  /// #. RAIIs necessary Ray resources such as object store, local scheduler.
  /// #. Initially distributes the cells to volumes via Plasma objects.
  ///    Each main volume will be accompanied by 6 + 12 + 8 = 26 halo (margin)
  ///    volumes. Each of the volume will have a determined ID.
  /// #. Put the number of steps and bounding box to the kSimulationStartMarker
  ///    object.
  /// #. Waits for the kSimulationEndMarker object.
  ///
  /// From the Python side, it will take the number of steps, construct a chain
  /// of remote calls to actually run each step based on the ID of the regions,
  /// and finally mark the end of the simulation.
  ///
  /// \param steps number of steps to simulate.
  virtual void Simulate(uint64_t steps) override;

  virtual ~RayScheduler() {
  }

 private:
  /// Establishes connections to Ray's local scheduler and Plasma object store.
  arrow::Status MaybeInitializeConnection();

  /// Stores `volumes` in the object store for `box` in `step`.
  arrow::Status StoreVolumes(
      long step,
      long box,
      const SurfaceToVolumeMap& volumes);

  void DisassembleResourceManager(
      ResourceManager<>* rm, const Partitioner* partitioner,
      int64_t step, int64_t box);

  /// Add all simulation objects from `box`'s `surface` in `step` to `rm`.
  arrow::Status AddFromVolume(ResourceManager<>* rm, long step, long box, Surface surface);

  /// Reassembles all volumes required to simulate `box` in `step` according to
  /// `partitioner`.
  ResourceManager<>* ReassembleVolumes(
      long step, long box, const Partitioner* partitioner);

  /// Calls Plasma `Fetch` and `Get` on `key`.
  std::vector<plasma::ObjectBuffer> FetchAndGetVolume(
      const plasma::ObjectID& key);

  /// Partitions cells into 3D volumes and their corresponding halo volumes.
  ///
  /// The results of the partitioning are stored in the object store directly.
  ///
  /// \param boundingBox output argument to receive the bounding box of the world
  virtual void InitiallyPartition(Box* boundingBox);

  bool initialized_ = false;
  std::unique_ptr<LocalSchedulerConnection> local_scheduler_ = nullptr;
  plasma::PlasmaClient object_store_;
};

class RaySimulation : public Simulation<> {
 public:
  using super = Simulation<>;
  RaySimulation();
  RaySimulation(int argc, const char **argv) : super(argc, argv) {}
  virtual ~RaySimulation() {}
  virtual Scheduler<Simulation>* GetScheduler() override {
    if (!scheduler_set_) {
      ReplaceScheduler(new RayScheduler());
      scheduler_set_ = true;
    }
    return super::GetScheduler();
  }
  virtual void ReplaceResourceManager(ResourceManager<>* rm) {
    rm_ = rm;
  }
 private:
  bool scheduler_set_ = false;
};

}  // namespace bdm

#endif  // DEMO_DISTRIBUTED_BDM_RAY_H_
