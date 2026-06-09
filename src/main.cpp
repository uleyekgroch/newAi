#include "agent/agent_kernel.hpp"
#include <iostream>
#include <iomanip>
#include <random>

namespace {

// A simple grid-world environment for demonstration.
// The agent observes a 1D "grid" and earns reward for reaching a target pattern.
class SimpleGridEnv final : public uik::IEnvironment {
public:
    explicit SimpleGridEnv(uik::Dim grid_size = 64, unsigned seed = 123)
        : grid_size_(grid_size), rng_(seed)
    {
        target_.assign(grid_size, 0.0);
        std::uniform_int_distribution<int> dist(0, 9);
        for (auto& v : target_) v = static_cast<uik::Real>(dist(rng_));
    }

    uik::Observation reset() override {
        state_.assign(grid_size_, 0.0);
        step_ = 0;
        return observe();
    }

    uik::StepResult step(const uik::Action& action) override {
        ++step_;
        // Actions: 0=noop, 1=shift-right, 2=shift-left, 3=increment
        switch (action.id) {
            case 1:
                std::rotate(state_.begin(), state_.begin() + 1, state_.end());
                break;
            case 2:
                std::rotate(state_.rbegin(), state_.rbegin() + 1, state_.rend());
                break;
            case 3:
                for (auto& v : state_) {
                    v = std::fmod(v + 1.0, 10.0);
                }
                break;
            default:
                break;
        }

        // Reward: negative distance to target
        uik::Real dist = 0.0;
        for (uik::Dim i = 0; i < grid_size_; ++i) {
            uik::Real d = state_[i] - target_[i];
            dist += d * d;
        }
        uik::Real reward = -std::sqrt(dist) / static_cast<uik::Real>(grid_size_);
        bool done = (step_ >= 200);

        return {observe(), reward, done};
    }

    int action_space_size() const override { return 4; }

private:
    uik::Dim grid_size_;
    std::mt19937 rng_;
    std::vector<uik::Real> state_;
    std::vector<uik::Real> target_;
    std::size_t step_ = 0;

    uik::Observation observe() const {
        return {uik::Tensor({grid_size_}, state_)};
    }
};

} // anonymous namespace

int main() {
    std::cout << "=== Universal Intelligence Kernel v0.1 ===" << std::endl;
    std::cout << "Architecture: WorldModel + SymbolicDescent + MetaEvolution"
              << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // Configure the agent kernel
    uik::agent::AgentKernel::Config config;
    config.wm_config.input_dim    = 64;
    config.wm_config.latent_dim   = 16;
    config.wm_config.action_space = 4;
    config.planning_horizon       = 5;
    config.evolve_interval        = 10;

    uik::agent::AgentKernel kernel(config);

    // Create environment
    SimpleGridEnv env(64, 42);

    // Run the agent
    constexpr std::size_t max_steps = 200;
    std::cout << "Running agent for " << max_steps << " steps..." << std::endl;
    kernel.run(env, max_steps);

    // Report results
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Steps completed: " << kernel.step_count() << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Total reward:    " << kernel.total_reward() << std::endl;
    std::cout << "Archive size:    " << kernel.evolution().archive().size()
              << std::endl;

    // Print last 10 step logs
    const auto& logs = kernel.logs();
    std::size_t start = logs.size() > 10 ? logs.size() - 10 : 0;
    std::cout << "\nLast steps:" << std::endl;
    std::cout << std::setw(6) << "Step"
              << std::setw(10) << "Novelty"
              << std::setw(12) << "CompProg"
              << std::setw(10) << "ExtRew"
              << std::setw(10) << "IntRew"
              << std::setw(8) << "Action" << std::endl;
    for (std::size_t i = start; i < logs.size(); ++i) {
        const auto& l = logs[i];
        std::cout << std::setw(6) << l.step
                  << std::setw(10) << l.novelty
                  << std::setw(12) << l.compression_prog
                  << std::setw(10) << l.external_reward
                  << std::setw(10) << l.intrinsic_reward
                  << std::setw(8) << l.action_id << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}
