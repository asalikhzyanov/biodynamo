#include "biology_module_op.h"
#include "biology_module_op_test.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "transactional_vector.h"

namespace bdm {
namespace biology_module_op_test_internal {

template <typename T>
void RunTest(T* cells) {
  MyCell<> cell_1(12);
  cell_1.AddBiologyModule(GrowthModule(2));

  MyCell<> cell_2(34);
  cell_2.AddBiologyModule(GrowthModule(3));

  cells->push_back(cell_1);
  cells->push_back(cell_2);

  BiologyModuleOp op;
  op.Compute(cells);

  EXPECT_EQ(2u, cells->size());
  EXPECT_NEAR(14, (*cells)[0].GetDiameter(), abs_error<double>::value);
  EXPECT_NEAR(37, (*cells)[1].GetDiameter(), abs_error<double>::value);
}

TEST(BiologyModuleOpTest, ComputeAos) {
  TransactionalVector<MyCell<Scalar>> cells;
  RunTest(&cells);
}

TEST(BiologyModuleOpTest, ComputeSoa) {
  auto cells = MyCell<>::NewEmptySoa();
  RunTest(&cells);
}

}  // namespace biology_module_op_test_internal
}  // namespace bdm
