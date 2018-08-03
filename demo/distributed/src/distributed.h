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

#ifndef DEMO_DISTRIBUTED_DISTRIBUTED_H_
#define DEMO_DISTRIBUTED_DISTRIBUTED_H_

#include <memory>

#include "biodynamo.h"
#include "common/event_loop.h"
#include "local_scheduler/local_scheduler_client.h"
#include "plasma/client.h"

extern std::string g_local_scheduler_socket_name;
extern std::string g_object_store_socket_name;
extern std::string g_object_store_manager_socket_name;

constexpr char kSimulationStartMarker[] = "aaaaaaaaaaaaaaaaaaaa";
constexpr char kSimulationEndMarker[] = "bbbbbbbbbbbbbbbbbbbb";

namespace bdm {

// -----------------------------------------------------------------------------
// This model creates a grid of 128x128x128 cells. Each cell grows untill a
// specific volume, after which it proliferates (i.e. divides).
// -----------------------------------------------------------------------------

// 1. Define compile time parameter
template <typename Backend>
struct CompileTimeParam : public DefaultCompileTimeParam<Backend> {
  // use predefined biology module GrowDivide
  using BiologyModules = Variant<GrowDivide>;
  // use default Backend and AtomicTypes
  using SimulationBackend = Scalar;
};

class Surface {
 public:
  Surface() : Surface(0) {}

  constexpr Surface intersect(const Surface& s) const {
    return Surface(value_ |  s.value_);
  }

  constexpr bool operator==(const Surface& other) const {
    return value_ == other.value_;
  }

  constexpr Surface operator|(const Surface& other) const {
    return intersect(other);
  }

  constexpr bool conflict(const Surface& other) const {
    return (value_ == 1 && other.value_ == 8) ||
        (value_ == 2 && other.value_ == 16) ||
        (value_ == 4 && other.value_ == 32) ||
        (value_ == 8 && other.value_ == 1) ||
        (value_ == 16 && other.value_ == 4) ||
        (value_ == 32 && other.value_ == 8);
  }

 private:
  constexpr explicit Surface(int v) : value_(v) {}
  int value_;
  friend class SurfaceEnum;
};

class SurfaceEnum {
 public:
  static constexpr Surface kNone{0};
  static constexpr Surface kLeft{1};
  static constexpr Surface kFront{2};
  static constexpr Surface kBottom{4};
  static constexpr Surface kRight{8};
  static constexpr Surface kBack{16};
  static constexpr Surface kTop{32};
};

class RayScheduler : public Scheduler<Simulation<>> {
 public:
  using super = Scheduler<Simulation<>>;

  /// Initiates a distributed simulation and waits for its completion.
  ///
  /// This method will:
  ///
  /// #. RAIIs necessary Ray resources such as object store, local scheduler.
  /// #. Initially distributes the cells to volumes via Plasma objects.
  ///    Each main volume will be accompanied by 6 + 12 = 18 halo (margin)
  ///    volumes. Each of the volume will have a determined ID.
  /// #. Put the number of steps to the kSimulationStartMarker object.
  /// #. Waits for the kSimulationEndMarker object.
  ///
  /// From the Python side, it will take the number of steps, construct a chain
  /// of remote calls to actually run each step based on the ID of the regions,
  /// and finally mark the end of the simulation.
  ///
  /// \param steps number of steps to simulate.
  virtual void Simulate(uint64_t steps) override {
    std::cout << "In RayScheduler::Simulate\n";
    local_scheduler_.reset(LocalSchedulerConnection_init(
        g_local_scheduler_socket_name.c_str(),
        UniqueID::from_random(),
        false,
        false
    ));
    if (!local_scheduler_) {
      std::cerr << "Cannot create new local scheduler connection to \""
                << g_local_scheduler_socket_name
                << "\". Simulation aborted.\n";
      return;
    }
    arrow::Status s = object_store_.Connect(
        g_object_store_socket_name.c_str(),
        g_object_store_manager_socket_name.c_str());
    if (!s.ok()) {
      std::cerr << "Cannot connect to object store (\""
                << g_object_store_socket_name
                << "\", \""
                << g_object_store_manager_socket_name
                << "\"). " << s << " Simulation aborted.\n";
      return;
    }
    std::shared_ptr<Buffer> buffer;
    s = object_store_.Create(plasma::ObjectID::from_binary(kSimulationStartMarker),
                             sizeof(steps),
                             nullptr,
                             0,
                             &buffer);
    if (!s.ok()) {
      std::cerr << "Cannot create simulation start marker. " << s <<
                   " Simulation aborted\n";
      return;
    }
    memcpy(buffer->mutable_data(), &steps, sizeof(steps));
    s = object_store_.Seal(plasma::ObjectID::from_binary(kSimulationStartMarker));
    if (!s.ok()) {
      std::cerr << "Cannot seal simulation start marker. " << s <<
                   "Simulation aborted\n";
      return;
    }
    s = object_store_.Release(plasma::ObjectID::from_binary(kSimulationStartMarker));
    if (!s.ok()) {
      std::cerr << "Cannot release simulation start marker. " << s <<
                   "Simulation aborted\n";
      return;
    }
    Partition();
    std::vector<plasma::ObjectBuffer> _ignored;
    std::cout << "Waiting for end of simulation...\n";
    s = object_store_.Get({plasma::ObjectID::from_binary(kSimulationEndMarker)},
                          -1,
                          &_ignored);
    if (!s.ok()) {
      std::cerr << "Error waiting for simulation end marker. " << s << '\n';
      return;
    }
  }

  using ResourceManagerPtr = std::shared_ptr<ResourceManager<>>;
  using SurfaceToVolume = std::pair<Surface, ResourceManagerPtr>;

  /// Allocates memory for the main volume, its 6 surfaces, and 12 edges.
  static std::array<SurfaceToVolume, 19> AllocVolumes() {
    const std::array<Surface, 6> surface_list =
        {SurfaceEnum::kLeft, SurfaceEnum::kFront, SurfaceEnum::kBottom,
         SurfaceEnum::kRight, SurfaceEnum::kBack, SurfaceEnum::kTop};
    std::array<SurfaceToVolume, 19> ret;
    ret[0].second.reset(new ResourceManager<>());
    size_t i = 1;
    for (size_t outer = 0; outer < surface_list.size(); ++outer) {
      const Surface &full_surface = surface_list[outer];
      ret[i].first = full_surface;
      ret[i].second.reset(new ResourceManager<>());
      ++i;
      for (size_t inner = outer + 1; inner < surface_list.size(); ++inner) {
        const Surface &adjacent_surface = surface_list[inner];
        if (full_surface.conflict(adjacent_surface)) {
          continue;
        }
        ret[i].first = full_surface | adjacent_surface;
        ret[i].second.reset(new ResourceManager<>());
        ++i;
      }
    }
    assert(i == ret.size());
    return ret;
  }

  /// Returns true if pos is in a bounded box.
  ///
  /// /param pos the location to check
  /// /param left_front_bottom (inclusive)
  /// /param right_back_top (exclusive)
  static inline bool is_in(const std::array<double, 3>& pos,
                           const std::array<double, 3>& left_front_bottom,
                           const std::array<double, 3>& right_back_top) {
    double x = pos[0];
    double y = pos[1];
    double z = pos[2];
    return x >= left_front_bottom[0] && x < right_back_top[0] &&
        y >= left_front_bottom[1] && y < right_back_top[1] &&
        z >= left_front_bottom[2] && z < right_back_top[2];
  }
  /// Returns a list of border surfaces that this_point belongs to.
  ///
  /// A point may belong to 1 to 3 of the 6 surfaces, and/or some of the 12
  /// edges (could be 0, 1, 2, or 3 edges).
  ///
  /// \param this_point the point location that we want to find surfaces for
  /// \param left_front_bottom one anchor of the box
  /// \param right_back_top the other anchor of the box
  /// \param xyz_halos the margins corresponding to x-, y-, and z-axis
  /// \return list of Surfaces, terminating in Surface::kNone
  static std::array<Surface, 6> FindContainingSurfaces(
      const std::array<double, 3>& this_point,
      const std::array<double, 3>& left_front_bottom,
      const std::array<double, 3>& right_back_top,
      const std::array<double, 3>& xyz_halos) {
    std::array<Surface, 6> ret;
    // Ensure that the halo is within the region.
    assert(xyz_halos[0] >= 0);
    assert(xyz_halos[1] >= 0);
    assert(xyz_halos[2] >= 0);
    assert(left_front_bottom[0] + xyz_halos[0] < right_back_top[0]);
    assert(left_front_bottom[1] + xyz_halos[1] < right_back_top[1]);
    assert(left_front_bottom[2] + xyz_halos[2] < right_back_top[2]);
    if (!is_in(this_point, left_front_bottom, right_back_top)) {
      return ret;
    }
    int i = 0;
    if (is_in(this_point, left_front_bottom, {
        right_back_top[0], left_front_bottom[1] + xyz_halos[1], right_back_top[2]})) {
      ret[i++] = SurfaceEnum::kFront;
    }
    if (is_in(this_point, left_front_bottom, {
        right_back_top[0], left_front_bottom[1] + xyz_halos[1], left_front_bottom[2] + xyz_halos[2]})) {
      ret[i++] = SurfaceEnum::kFront | SurfaceEnum::kBottom;
    }
    if (is_in(this_point, left_front_bottom, {
        left_front_bottom[0] + xyz_halos[0], left_front_bottom[1] + xyz_halos[1], right_back_top[2]})) {
      ret[i++] = SurfaceEnum::kFront | SurfaceEnum::kLeft;
    }
    if (is_in(this_point, {left_front_bottom[0], left_front_bottom[1], right_back_top[2] - xyz_halos[2]},
              {right_back_top[0], left_front_bottom[1] + xyz_halos[1], right_back_top[2]})) {
      ret[i++] = SurfaceEnum::kFront | SurfaceEnum::kTop;
    }
    if (is_in(this_point, {right_back_top[0] - xyz_halos[0], left_front_bottom[1], left_front_bottom[2]},
              {right_back_top[0], left_front_bottom[1] + xyz_halos[1], right_back_top[2]})) {
      ret[i++] = SurfaceEnum::kFront | SurfaceEnum::kRight;
    }
    if (is_in(this_point, {left_front_bottom[0], left_front_bottom[1], right_back_top[2] - xyz_halos[2]},
              {right_back_top})) {
      ret[i++] = SurfaceEnum::kTop;
    }
    if (is_in(this_point, {left_front_bottom[0], left_front_bottom[1], right_back_top[2] - xyz_halos[2]},
              {left_front_bottom[0] + xyz_halos[0], right_back_top[1], right_back_top[2]})) {
      ret[i++] = SurfaceEnum::kTop | SurfaceEnum::kLeft;
    }
    if (is_in(this_point, {right_back_top[0] - xyz_halos[0], left_front_bottom[1], right_back_top[2] - xyz_halos[2]},
              right_back_top)) {
      ret[i++] = SurfaceEnum::kTop | SurfaceEnum::kRight;
    }
    if (is_in(this_point, {left_front_bottom[0], right_back_top[1] - xyz_halos[1], right_back_top[2] - xyz_halos[2]},
              right_back_top)) {
      ret[i++] = SurfaceEnum::kTop | SurfaceEnum::kBack;
    }
    if (is_in(this_point, {left_front_bottom[0], right_back_top[1] - xyz_halos[1], left_front_bottom[2]},
              right_back_top)) {
      ret[i++] = SurfaceEnum::kBack;
    }
    if (is_in(this_point, {left_front_bottom[0], right_back_top[1] - xyz_halos[1], left_front_bottom[2]},
              {left_front_bottom[0] + xyz_halos[0], right_back_top[1], right_back_top[2]})) {
      ret[i++] = SurfaceEnum::kBack | SurfaceEnum::kLeft;
    }
    if (is_in(this_point, {right_back_top[0] - xyz_halos[0], right_back_top[1] - xyz_halos[1], left_front_bottom[2]},
              right_back_top)) {
      ret[i++] = SurfaceEnum::kBack | SurfaceEnum::kRight;
    }
    if (is_in(this_point, {left_front_bottom[0], right_back_top[1] - xyz_halos[1], left_front_bottom[2]},
              {right_back_top[0], right_back_top[1], left_front_bottom[2] + xyz_halos[2]})) {
      ret[i++] = SurfaceEnum::kBack | SurfaceEnum::kBottom;
    }
    if (is_in(this_point, left_front_bottom,
              {right_back_top[0], right_back_top[1], left_front_bottom[2] + xyz_halos[2]})) {
      ret[i++] = SurfaceEnum::kBottom;
    }
    if (is_in(this_point, left_front_bottom,
              {left_front_bottom[0] + xyz_halos[0], right_back_top[1], left_front_bottom[2] + xyz_halos[2]})) {
      ret[i++] = SurfaceEnum::kBottom | SurfaceEnum::kLeft;
    }
    if (is_in(this_point, {right_back_top[0] - xyz_halos[0], left_front_bottom[1], left_front_bottom[2]},
              {right_back_top[0], right_back_top[1], left_front_bottom[2] + xyz_halos[2]})) {
      ret[i++] = SurfaceEnum::kBottom | SurfaceEnum::kRight;
    }
    if (is_in(this_point, left_front_bottom,
              {left_front_bottom[0] + xyz_halos[0], right_back_top[1], right_back_top[2]})) {
      ret[i++] = SurfaceEnum::kLeft;
    }
    if (is_in(this_point, {right_back_top[0] - xyz_halos[0], left_front_bottom[1], left_front_bottom[2]},
              right_back_top)) {
      ret[i++] = SurfaceEnum::kRight;
    }
    assert(i <= 6);
    return ret;
  }

  /// Returns the ResourceManager for the specified surface.
  static ResourceManagerPtr FindResourceManager(
      const std::array<SurfaceToVolume, 19>& map,
      Surface s) {
    return nullptr;
  }

  std::array<SurfaceToVolume, 19> CreateVolumesForBox(
      ResourceManager<>* rm,
      const std::array<double, 3>& left_front_bottom,
      const std::array<double, 3>& right_back_top) {
    std::array<SurfaceToVolume, 19> ret = AllocVolumes();
    auto f = [&](const auto& element, bdm::SoHandle) {
      std::array<double, 3> pos = element.GetPosition();
      if (is_in(pos, left_front_bottom, right_back_top)) {
        ResourceManagerPtr node = ret[0].second;
        node->push_back(element);
        for (Surface s : FindContainingSurfaces(
            pos, left_front_bottom, right_back_top, {1, 1, 1})) {
          if (s == SurfaceEnum::kNone) {
            break;
          }
          ResourceManagerPtr sub_rm = FindResourceManager(ret, s);
          //sub_rm->push_back(element);
        }
      }
    };
    rm->ApplyOnAllElements(f);
    return ret;
  };

  /// Partitions cells into 3D volumes and their corresponding halo volumes.
  ///
  /// This should be a separate class so that the partitioning logic is
  /// independent of the scheduler.
  ///
  /// The results of the partitioning are stored in the object store directly.
  virtual void Partition() {
    std::cout << "In RayScheduler::Partition\n";
    Simulation<> *sim = Simulation<>::GetActive();
    ResourceManager<> *rm = sim->GetResourceManager();

    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::min();
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::min();
    auto f = [&](auto element, bdm::SoHandle) {
      std::array<double, 3> pos = element.GetPosition();
      min_x = std::min(min_x, pos[0]);
      max_x = std::max(max_x, pos[0]);
      min_y = std::min(min_y, pos[1]);
      max_y = std::max(max_y, pos[1]);
      min_z = std::min(min_z, pos[2]);
      max_z = std::max(max_z, pos[2]);
    };
    rm->ApplyOnAllElements(f);
    max_x += 1e-9;
    max_y += 1e-9;
    max_z += 1e-9;

    std::cout << "min\tmax\n";
    std::cout << min_x << '\t' << max_x << '\n';
    std::cout << min_y << '\t' << max_y << '\n';
    std::cout << min_z << '\t' << max_z << '\n';

    double mid_x = min_x + (max_x - min_x) / 2;
    std::array<SurfaceToVolume, 19> node_1 = CreateVolumesForBox(
        rm,
        {min_x, min_y, min_z},
        {mid_x, max_y, max_z});
    std::array<SurfaceToVolume, 19> node_2 = CreateVolumesForBox(
        rm,
        {mid_x, min_y, min_z},
        {max_x, max_y, max_z});
    std::cout << "Total " << rm->GetNumSimObjects() << '\n';
    std::cout << "Node 1 " << node_1[0].second->GetNumSimObjects() << '\n';
    std::cout << "Node 2 " << node_2[0].second->GetNumSimObjects() << '\n';
  }

  virtual ~RayScheduler() {
  }

 private:
  std::unique_ptr<LocalSchedulerConnection> local_scheduler_ = nullptr;
  plasma::PlasmaClient object_store_;
};

class RaySimulation : public Simulation<> {
 public:
  using super = Simulation<>;
  RaySimulation(int argc, const char **argv) : super(argc, argv) {}
  virtual ~RaySimulation() {}
  virtual Scheduler<Simulation>* GetScheduler() override {
    if (!scheduler_set_) {
      ReplaceScheduler(new RayScheduler());
      scheduler_set_ = true;
    }
    return super::GetScheduler();
  }
 private:
  bool scheduler_set_ = false;
};

inline int Simulate(int argc, const char** argv) {
  // 2. Create new simulation
  RaySimulation simulation(argc, argv);

  // 3. Define initial model - in this example: 3D grid of cells
  size_t cells_per_dim = 128;
  auto construct = [](const std::array<double, 3> &position) {
    Cell cell(position);
    cell.SetDiameter(30);
    cell.SetAdherence(0.4);
    cell.SetMass(1.0);
    cell.AddBiologyModule(GrowDivide());
    return cell;
  };
  ModelInitializer::Grid3D(cells_per_dim, 20, construct);

  // 4. Run simulation for one timestep
  simulation.GetScheduler()->Simulate(42);

  std::cout << "Simulation completed successfully!\n";
  return 0;
}

}  // namespace bdm
#endif  // DEMO_DISTRIBUTED_DISTRIBUTED_H_
