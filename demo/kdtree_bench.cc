#include "cell.h"
#include "displacement_op.h"
#include "inline_vector.h"
#include "neighbor_nanoflann_op.h"
#include "neighbor_op.h"
#include "neighbor_pcl_op.h"
#include "neighbor_unibn_op.h"

#include <chrono>
#include <fstream>

#include <sys/stat.h>
#include <sys/types.h>

using std::ofstream;
using bdm::Cell;
using bdm::NeighborNanoflannOp;
using bdm::DisplacementOp;
using bdm::Scalar;

template <typename T, typename NOp, typename DOp>
void RunTest(T* cells, const NOp& nop, const DOp& dop) {
  // execute and time neighbor operation
  std::string threads;
  if (std::getenv("OMP_NUM_THREADS")) {
    threads = std::string(std::getenv("OMP_NUM_THREADS"));
  } else {
    std::cout << "You didn't set the number of threads with OMP_NUM_THREADS!" << std::endl;
    exit(1);
  }
  omp_set_num_threads(std::stoi(threads));
  std::cout << cells->size() << "," << threads << ",";
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  nop.Compute(cells);
  dop.Compute(cells);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
}

int main(int args, char** argv) {
  if(args == 2) {
    size_t cells_per_dim;
    std::istringstream(std::string(argv[1])) >> cells_per_dim;

    srand(4357);

    auto cells = Cell<>::NewEmptySoa();
    cells.reserve(cells_per_dim * cells_per_dim * cells_per_dim);

    const double space = 20;
    for (size_t i = 0; i < cells_per_dim; i++) {
      for (size_t j = 0; j < cells_per_dim; j++) {
        for (size_t k = 0; k < cells_per_dim; k++) {
          Cell<Scalar> cell({i * space, j * space, k * space});
          cell.SetDiameter(30);
          cell.SetAdherence(0.4);
          cell.SetMass(1.0);
          cell.UpdateVolume();
          cells.push_back(cell);
        }
      }
    }

    std::cout << "- This line is to replace upstream initialization message -" << std::endl;
    std::cout << "Number of simulation objects,Number of threads,Physics time" << std::endl;
    RunTest(&cells, NeighborNanoflannOp(900), DisplacementOp());
  } else {
    std::cout << "Error args: ./octree_bench <cells_per_dim>" << std::endl;
  }

  return 0;
}
