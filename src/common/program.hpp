#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <sstream>
#include <functional>

namespace uik {

// ── Program AST ──
// A program is a tree of operations in a minimal DSL.
// Uses std::variant for type-safe sum type (C++20 pattern matching style).

enum class OpKind {
    Identity,     // no-op: return input
    Constant,     // produce a fixed value
    Rotate90,     // rotate grid 90°
    FlipH,        // horizontal flip
    FlipV,        // vertical flip
    Translate,    // shift by (dx, dy)
    Fill,         // fill region with value
    MapColor,     // map one color to another
    Compose,      // sequential composition of two programs
    Conditional,  // if-then-else based on a predicate
};

struct ProgramNode;
using ProgramPtr = std::shared_ptr<ProgramNode>;

struct ProgramNode {
    OpKind kind;
    int    param1 = 0;
    int    param2 = 0;
    std::vector<ProgramPtr> children;

    [[nodiscard]] std::size_t description_length() const {
        std::size_t len = 1; // 1 for the opcode
        if (kind == OpKind::Translate || kind == OpKind::Fill ||
            kind == OpKind::MapColor || kind == OpKind::Constant) {
            len += 2; // params cost
        }
        for (const auto& child : children) {
            len += child->description_length();
        }
        return len;
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        print(oss, 0);
        return oss.str();
    }

    [[nodiscard]] std::size_t depth() const {
        std::size_t max_child = 0;
        for (const auto& c : children) {
            max_child = std::max(max_child, c->depth());
        }
        return 1 + max_child;
    }

    [[nodiscard]] std::size_t node_count() const {
        std::size_t count = 1;
        for (const auto& c : children) {
            count += c->node_count();
        }
        return count;
    }

private:
    void print(std::ostringstream& oss, int indent) const {
        for (int i = 0; i < indent; ++i) oss << "  ";
        oss << op_name();
        if (kind == OpKind::Translate) {
            oss << "(" << param1 << "," << param2 << ")";
        } else if (kind == OpKind::MapColor || kind == OpKind::Fill) {
            oss << "(" << param1 << "->" << param2 << ")";
        } else if (kind == OpKind::Constant) {
            oss << "(" << param1 << ")";
        }
        oss << "\n";
        for (const auto& c : children) {
            c->print(oss, indent + 1);
        }
    }

    [[nodiscard]] std::string op_name() const {
        switch (kind) {
            case OpKind::Identity:    return "Id";
            case OpKind::Constant:    return "Const";
            case OpKind::Rotate90:    return "Rot90";
            case OpKind::FlipH:       return "FlipH";
            case OpKind::FlipV:       return "FlipV";
            case OpKind::Translate:   return "Trans";
            case OpKind::Fill:        return "Fill";
            case OpKind::MapColor:    return "MapCol";
            case OpKind::Compose:     return "Compose";
            case OpKind::Conditional: return "If";
        }
        return "?";
    }
};

// Factory helpers
inline ProgramPtr make_program(OpKind kind, int p1 = 0, int p2 = 0,
                                std::vector<ProgramPtr> children = {}) {
    return std::make_shared<ProgramNode>(
        ProgramNode{kind, p1, p2, std::move(children)});
}

inline ProgramPtr identity() {
    return make_program(OpKind::Identity);
}

inline ProgramPtr compose(ProgramPtr first, ProgramPtr second) {
    return make_program(OpKind::Compose, 0, 0, {std::move(first), std::move(second)});
}

} // namespace uik
