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
    // Grid transform primitives (Phase 1)
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

    // General computation primitives (Phase 3)
    Add,          // elementwise addition (param1 = scalar offset)
    Multiply,     // elementwise multiplication (param1 = scalar factor)
    Modulo,       // elementwise modulo by param1
    Threshold,    // if cell > param1: param2, else 0
    Count,        // count occurrences of param1, broadcast result
    Repeat,       // apply child[0] param1 times (loop)
    Fold,         // reduce: apply child[0] across elements
    Zip,          // combine two children elementwise
    Filter,       // keep only elements matching param1
    Store,        // variable store: save result to slot param1
    Load,         // variable load: recall from slot param1
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
            kind == OpKind::MapColor || kind == OpKind::Constant ||
            kind == OpKind::Add || kind == OpKind::Multiply ||
            kind == OpKind::Modulo || kind == OpKind::Threshold ||
            kind == OpKind::Count || kind == OpKind::Repeat ||
            kind == OpKind::Filter || kind == OpKind::Store ||
            kind == OpKind::Load) {
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
            case OpKind::Add:         return "Add";
            case OpKind::Multiply:    return "Mul";
            case OpKind::Modulo:      return "Mod";
            case OpKind::Threshold:   return "Thresh";
            case OpKind::Count:       return "Count";
            case OpKind::Repeat:      return "Repeat";
            case OpKind::Fold:        return "Fold";
            case OpKind::Zip:         return "Zip";
            case OpKind::Filter:      return "Filter";
            case OpKind::Store:       return "Store";
            case OpKind::Load:        return "Load";
        }
        return "?";
    }
};

// ── Serialization ──
// Format: "(OpName p1 p2 child1 child2 ...)"
// Example: "(Compose 0 0 (FlipH 0 0) (MapCol 3 5))"

[[nodiscard]] inline std::string serialize(const ProgramPtr& prog) {
    if (!prog) return "()";
    auto op_str = [](OpKind k) -> std::string {
        switch (k) {
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
            case OpKind::Add:         return "Add";
            case OpKind::Multiply:    return "Mul";
            case OpKind::Modulo:      return "Mod";
            case OpKind::Threshold:   return "Thresh";
            case OpKind::Count:       return "Count";
            case OpKind::Repeat:      return "Repeat";
            case OpKind::Fold:        return "Fold";
            case OpKind::Zip:         return "Zip";
            case OpKind::Filter:      return "Filter";
            case OpKind::Store:       return "Store";
            case OpKind::Load:        return "Load";
        }
        return "?";
    };
    std::ostringstream oss;
    oss << "(" << op_str(prog->kind) << " " << prog->param1 << " " << prog->param2;
    for (const auto& c : prog->children) {
        oss << " " << serialize(c);
    }
    oss << ")";
    return oss.str();
}

[[nodiscard]] inline ProgramPtr deserialize(const std::string& s) {
    std::size_t pos = 0;

    std::function<ProgramPtr()> parse = [&]() -> ProgramPtr {
        while (pos < s.size() && s[pos] == ' ') ++pos;
        if (pos >= s.size() || s[pos] != '(') return nullptr;
        ++pos; // skip '('

        // read op name
        std::string name;
        while (pos < s.size() && s[pos] != ' ' && s[pos] != ')') {
            name += s[pos++];
        }
        if (name.empty()) { // "()"
            if (pos < s.size() && s[pos] == ')') ++pos;
            return nullptr;
        }

        auto str_to_op = [](const std::string& n) -> OpKind {
            if (n == "Id")      return OpKind::Identity;
            if (n == "Const")   return OpKind::Constant;
            if (n == "Rot90")   return OpKind::Rotate90;
            if (n == "FlipH")   return OpKind::FlipH;
            if (n == "FlipV")   return OpKind::FlipV;
            if (n == "Trans")   return OpKind::Translate;
            if (n == "Fill")    return OpKind::Fill;
            if (n == "MapCol")  return OpKind::MapColor;
            if (n == "Compose") return OpKind::Compose;
            if (n == "If")      return OpKind::Conditional;
            if (n == "Add")     return OpKind::Add;
            if (n == "Mul")     return OpKind::Multiply;
            if (n == "Mod")     return OpKind::Modulo;
            if (n == "Thresh")  return OpKind::Threshold;
            if (n == "Count")   return OpKind::Count;
            if (n == "Repeat")  return OpKind::Repeat;
            if (n == "Fold")    return OpKind::Fold;
            if (n == "Zip")     return OpKind::Zip;
            if (n == "Filter")  return OpKind::Filter;
            if (n == "Store")   return OpKind::Store;
            if (n == "Load")    return OpKind::Load;
            return OpKind::Identity;
        };

        OpKind kind = str_to_op(name);

        // read p1
        while (pos < s.size() && s[pos] == ' ') ++pos;
        int p1 = 0;
        {
            bool neg = false;
            if (pos < s.size() && s[pos] == '-') { neg = true; ++pos; }
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
                p1 = p1 * 10 + (s[pos++] - '0');
            }
            if (neg) p1 = -p1;
        }

        // read p2
        while (pos < s.size() && s[pos] == ' ') ++pos;
        int p2 = 0;
        {
            bool neg = false;
            if (pos < s.size() && s[pos] == '-') { neg = true; ++pos; }
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
                p2 = p2 * 10 + (s[pos++] - '0');
            }
            if (neg) p2 = -p2;
        }

        // read children
        std::vector<ProgramPtr> children;
        while (pos < s.size()) {
            while (pos < s.size() && s[pos] == ' ') ++pos;
            if (pos >= s.size() || s[pos] == ')') break;
            if (s[pos] == '(') {
                auto child = parse();
                if (child) children.push_back(std::move(child));
            } else {
                break;
            }
        }

        if (pos < s.size() && s[pos] == ')') ++pos;
        return std::make_shared<ProgramNode>(ProgramNode{kind, p1, p2, std::move(children)});
    };

    return parse();
}

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
