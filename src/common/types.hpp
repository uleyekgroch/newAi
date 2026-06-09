#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace uik {

using Real = double;
using Dim  = std::size_t;

// ── Tensor: lightweight N-d array (value semantics) ──
class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<Dim> shape, Real fill = 0.0)
        : shape_(std::move(shape))
    {
        Dim total = flat_size();
        data_.assign(total, fill);
    }

    Tensor(std::vector<Dim> shape, std::vector<Real> data)
        : shape_(std::move(shape)), data_(std::move(data))
    {
        if (data_.size() != flat_size()) {
            throw std::invalid_argument("Tensor data size mismatch with shape");
        }
    }

    [[nodiscard]] const std::vector<Dim>& shape() const { return shape_; }
    [[nodiscard]] Dim flat_size() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(),
                               Dim{1}, std::multiplies<>{});
    }

    [[nodiscard]] Real& at(Dim i) { return data_.at(i); }
    [[nodiscard]] Real  at(Dim i) const { return data_.at(i); }

    [[nodiscard]] Real* data() { return data_.data(); }
    [[nodiscard]] const Real* data() const { return data_.data(); }
    [[nodiscard]] Dim size() const { return data_.size(); }

    [[nodiscard]] bool empty() const { return data_.empty(); }

    Real dot(const Tensor& other) const {
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument("Tensor dot: size mismatch");
        }
        Real sum = 0.0;
        for (Dim i = 0; i < data_.size(); ++i) {
            sum += data_[i] * other.data_[i];
        }
        return sum;
    }

    Real l2_norm() const {
        Real sum = 0.0;
        for (auto v : data_) sum += v * v;
        return std::sqrt(sum);
    }

    Tensor operator-(const Tensor& other) const {
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument("Tensor sub: size mismatch");
        }
        Tensor result(shape_);
        for (Dim i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    Tensor operator+(const Tensor& other) const {
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument("Tensor add: size mismatch");
        }
        Tensor result(shape_);
        for (Dim i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor operator*(Real scalar) const {
        Tensor result(shape_);
        for (Dim i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    bool operator==(const Tensor& other) const {
        return shape_ == other.shape_ && data_ == other.data_;
    }

private:
    std::vector<Dim>  shape_;
    std::vector<Real> data_;
};

// ── Domain Value Objects ──

struct Observation {
    Tensor data;
};

struct State {
    Tensor latent;
};

struct Action {
    int id = 0;
};

struct Reward {
    Real external  = 0.0;
    Real intrinsic = 0.0;
    [[nodiscard]] Real total(Real alpha = 0.5) const {
        return external + alpha * intrinsic;
    }
};

struct StepResult {
    Observation observation;
    Real        reward   = 0.0;
    bool        done     = false;
};

} // namespace uik
