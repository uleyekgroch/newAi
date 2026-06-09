#include <gtest/gtest.h>
#include "symbolic_descent/dsl.hpp"

using namespace uik;
using namespace uik::symbolic_descent;

class ExtendedDSLTest : public ::testing::Test {
protected:
    DSL dsl_;
    Tensor input_{std::vector<Dim>{4}, std::vector<Real>{1, 2, 3, 4}};
};

TEST_F(ExtendedDSLTest, AddOffset) {
    auto prog = make_program(OpKind::Add, 10);
    auto result = dsl_.execute(prog, input_);
    EXPECT_DOUBLE_EQ(result.at(0), 11.0);
    EXPECT_DOUBLE_EQ(result.at(3), 14.0);
}

TEST_F(ExtendedDSLTest, Multiply) {
    auto prog = make_program(OpKind::Multiply, 3);
    auto result = dsl_.execute(prog, input_);
    EXPECT_DOUBLE_EQ(result.at(0), 3.0);
    EXPECT_DOUBLE_EQ(result.at(1), 6.0);
}

TEST_F(ExtendedDSLTest, Modulo) {
    auto prog = make_program(OpKind::Modulo, 3);
    auto result = dsl_.execute(prog, input_);
    EXPECT_DOUBLE_EQ(result.at(0), 1.0);
    EXPECT_DOUBLE_EQ(result.at(1), 2.0);
    EXPECT_DOUBLE_EQ(result.at(2), 0.0);
    EXPECT_DOUBLE_EQ(result.at(3), 1.0);
}

TEST_F(ExtendedDSLTest, ModuloByZeroNoOp) {
    auto prog = make_program(OpKind::Modulo, 0);
    auto result = dsl_.execute(prog, input_);
    EXPECT_DOUBLE_EQ(result.at(0), 1.0);
}

TEST_F(ExtendedDSLTest, Threshold) {
    auto prog = make_program(OpKind::Threshold, 2, 9);
    auto result = dsl_.execute(prog, input_);
    EXPECT_DOUBLE_EQ(result.at(0), 0.0);  // 1 <= 2
    EXPECT_DOUBLE_EQ(result.at(1), 0.0);  // 2 <= 2
    EXPECT_DOUBLE_EQ(result.at(2), 9.0);  // 3 > 2
    EXPECT_DOUBLE_EQ(result.at(3), 9.0);  // 4 > 2
}

TEST_F(ExtendedDSLTest, Count) {
    Tensor data({5}, std::vector<Real>{1, 2, 1, 3, 1});
    auto prog = make_program(OpKind::Count, 1);
    auto result = dsl_.execute(prog, data);
    EXPECT_DOUBLE_EQ(result.at(0), 3.0);
    EXPECT_DOUBLE_EQ(result.at(4), 3.0);
}

TEST_F(ExtendedDSLTest, Filter) {
    Tensor data({5}, std::vector<Real>{1, 2, 1, 3, 1});
    auto prog = make_program(OpKind::Filter, 1);
    auto result = dsl_.execute(prog, data);
    EXPECT_DOUBLE_EQ(result.at(0), 1.0);
    EXPECT_DOUBLE_EQ(result.at(1), 0.0);
    EXPECT_DOUBLE_EQ(result.at(2), 1.0);
}

TEST_F(ExtendedDSLTest, RepeatLoop) {
    auto prog = make_program(OpKind::Repeat, 3, 0,
                              {make_program(OpKind::Add, 1)});
    Tensor data({2}, std::vector<Real>{0, 0});
    auto result = dsl_.execute(prog, data);
    EXPECT_DOUBLE_EQ(result.at(0), 3.0);
}

TEST_F(ExtendedDSLTest, RepeatZeroTimes) {
    auto prog = make_program(OpKind::Repeat, 0, 0,
                              {make_program(OpKind::Add, 1)});
    Tensor data({2}, std::vector<Real>{5, 5});
    auto result = dsl_.execute(prog, data);
    EXPECT_DOUBLE_EQ(result.at(0), 5.0);
}

TEST_F(ExtendedDSLTest, Fold) {
    // Fold with Add child — effectively sums (each step adds pair[0]+pair[1])
    auto prog = make_program(OpKind::Fold, 0, 0,
                              {make_program(OpKind::Add, 0)});
    Tensor data({3}, std::vector<Real>{1, 2, 3});
    auto result = dsl_.execute(prog, data);
    // Fold applies child to [acc, element] pairs — Add(0) returns same values
    EXPECT_GT(result.at(0), 0.0);
}

TEST_F(ExtendedDSLTest, Zip) {
    auto prog = make_program(OpKind::Zip, 0, 0, {
        make_program(OpKind::Add, 1),
        make_program(OpKind::Multiply, 2)
    });
    Tensor data({3}, std::vector<Real>{1, 2, 3});
    auto result = dsl_.execute(prog, data);
    // (1+1)+(1*2)=4, (2+1)+(2*2)=7, (3+1)+(3*2)=10
    EXPECT_DOUBLE_EQ(result.at(0), 4.0);
    EXPECT_DOUBLE_EQ(result.at(1), 7.0);
    EXPECT_DOUBLE_EQ(result.at(2), 10.0);
}

TEST_F(ExtendedDSLTest, StoreAndLoad) {
    // Store to slot 0, then Load from slot 0
    auto prog = compose(
        make_program(OpKind::Store, 0),
        compose(
            make_program(OpKind::Add, 100),
            make_program(OpKind::Load, 0)
        )
    );
    Tensor data({2}, std::vector<Real>{5, 10});
    auto result = dsl_.execute(prog, data);
    // Store saves [5,10], Add makes [105,110], Load recalls [5,10]
    EXPECT_DOUBLE_EQ(result.at(0), 5.0);
    EXPECT_DOUBLE_EQ(result.at(1), 10.0);
}

TEST_F(ExtendedDSLTest, ExtendedPrimitivesIncludesNewOps) {
    auto prims = DSL::extended_primitives();
    EXPECT_GT(prims.size(), DSL::primitives().size());
}

TEST_F(ExtendedDSLTest, SerializeDeserializeNewOps) {
    auto prog = make_program(OpKind::Add, 5);
    auto serialized = uik::serialize(prog);
    auto deserialized = uik::deserialize(serialized);
    EXPECT_EQ(deserialized->kind, OpKind::Add);
    EXPECT_EQ(deserialized->param1, 5);
}
