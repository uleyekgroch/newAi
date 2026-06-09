#include <gtest/gtest.h>
#include "common/program.hpp"

using namespace uik;

TEST(Serialization, identity_roundtrip) {
    auto prog = identity();
    std::string s = serialize(prog);
    EXPECT_EQ(s, "(Id 0 0)");

    auto restored = deserialize(s);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->kind, OpKind::Identity);
}

TEST(Serialization, leaf_with_params) {
    auto prog = make_program(OpKind::Translate, 3, -2);
    std::string s = serialize(prog);
    EXPECT_EQ(s, "(Trans 3 -2)");

    auto restored = deserialize(s);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->kind, OpKind::Translate);
    EXPECT_EQ(restored->param1, 3);
    EXPECT_EQ(restored->param2, -2);
}

TEST(Serialization, compose_roundtrip) {
    auto prog = compose(
        make_program(OpKind::FlipH),
        make_program(OpKind::Rotate90)
    );
    std::string s = serialize(prog);
    EXPECT_EQ(s, "(Compose 0 0 (FlipH 0 0) (Rot90 0 0))");

    auto restored = deserialize(s);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->kind, OpKind::Compose);
    ASSERT_EQ(restored->children.size(), 2u);
    EXPECT_EQ(restored->children[0]->kind, OpKind::FlipH);
    EXPECT_EQ(restored->children[1]->kind, OpKind::Rotate90);
}

TEST(Serialization, nested_compose) {
    auto prog = compose(
        compose(make_program(OpKind::FlipH), make_program(OpKind::FlipV)),
        make_program(OpKind::MapColor, 3, 5)
    );
    std::string s = serialize(prog);
    auto restored = deserialize(s);

    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->kind, OpKind::Compose);
    ASSERT_EQ(restored->children.size(), 2u);
    EXPECT_EQ(restored->children[0]->kind, OpKind::Compose);
    EXPECT_EQ(restored->children[1]->kind, OpKind::MapColor);
    EXPECT_EQ(restored->children[1]->param1, 3);
    EXPECT_EQ(restored->children[1]->param2, 5);
}

TEST(Serialization, null_program) {
    std::string s = serialize(nullptr);
    EXPECT_EQ(s, "()");
    auto restored = deserialize(s);
    EXPECT_EQ(restored, nullptr);
}

TEST(Serialization, all_op_kinds) {
    auto ops = {OpKind::Identity, OpKind::Constant, OpKind::Rotate90,
                OpKind::FlipH, OpKind::FlipV, OpKind::Translate,
                OpKind::Fill, OpKind::MapColor, OpKind::Conditional};
    for (auto op : ops) {
        auto prog = make_program(op, 1, 2);
        auto s = serialize(prog);
        auto restored = deserialize(s);
        ASSERT_NE(restored, nullptr) << "Failed for op " << static_cast<int>(op);
        EXPECT_EQ(restored->kind, op);
        EXPECT_EQ(restored->param1, 1);
        EXPECT_EQ(restored->param2, 2);
    }
}

TEST(Serialization, description_length_preserved) {
    auto prog = compose(make_program(OpKind::FlipH), make_program(OpKind::Constant, 5));
    auto s = serialize(prog);
    auto restored = deserialize(s);
    EXPECT_EQ(restored->description_length(), prog->description_length());
}
