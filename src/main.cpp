#include "agent/agent_kernel.hpp"
#include "agent/grid_world.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Universal Intelligence Kernel v0.2 ===" << std::endl;
    std::cout << "Architecture: WorldModel + SymbolicDescent + MetaEvolution"
              << std::endl;
    std::cout << "Core loop: perceive -> model -> induce rules -> "
                 "set goal -> plan -> act -> evolve"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // ── 1. 2D GridWorld demo ──
    std::cout << "\n[Demo 1] 2D GridWorld (5x5, target = FlipH + Increment)\n";
    {
        uik::agent::GridWorld::Config env_cfg;
        env_cfg.rows = 5;
        env_cfg.cols = 5;
        env_cfg.max_steps = 200;

        uik::agent::AgentKernel::Config cfg;
        cfg.wm_config.input_dim    = env_cfg.rows * env_cfg.cols * 2; // grid + target
        cfg.wm_config.latent_dim   = 16;
        cfg.wm_config.action_space = 6;
        cfg.planning_horizon       = 4;
        cfg.evolve_interval        = 10;
        cfg.induce_interval        = 15;
        cfg.obs_buffer_size        = 8;
        cfg.planner_config.action_space = 6;

        uik::agent::AgentKernel kernel(cfg);
        uik::agent::GridWorld env(env_cfg);
        kernel.run(env, 200);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Steps:         " << kernel.step_count() << std::endl;
        std::cout << "  Total reward:  " << kernel.total_reward() << std::endl;
        std::cout << "  Rules induced: " << kernel.rule_library().size() << std::endl;
        std::cout << "  Archive size:  " << kernel.evolution().archive().size()
                  << std::endl;
        std::cout << "  Error rate:    " << kernel.world_model().prediction_error_rate()
                  << std::endl;

        const auto& logs = kernel.logs();
        std::size_t start = logs.size() > 5 ? logs.size() - 5 : 0;
        std::cout << "\n  Last steps:" << std::endl;
        std::cout << "  " << std::setw(6) << "Step"
                  << std::setw(10) << "Novelty"
                  << std::setw(10) << "CompProg"
                  << std::setw(10) << "ExtRew"
                  << std::setw(10) << "IntRew"
                  << std::setw(8) << "Action"
                  << std::setw(8) << "Rules" << std::endl;
        for (std::size_t i = start; i < logs.size(); ++i) {
            const auto& l = logs[i];
            std::cout << "  " << std::setw(6) << l.step
                      << std::setw(10) << l.novelty
                      << std::setw(10) << l.compression_prog
                      << std::setw(10) << l.external_reward
                      << std::setw(10) << l.intrinsic_reward
                      << std::setw(8) << l.action_id
                      << std::setw(8) << l.rules_count << std::endl;
        }
    }

    // ── 2. Simple 1D exploration demo ──
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "[Demo 2] 1D Exploration (64-dim, curiosity-driven)\n";
    {
        // Simple 1D environment inline
        class SimpleEnv final : public uik::IEnvironment {
        public:
            uik::Observation reset() override {
                step_ = 0;
                data_.assign(64, 0.0);
                return {uik::Tensor({64}, data_)};
            }
            uik::StepResult step(const uik::Action& action) override {
                ++step_;
                switch (action.id) {
                    case 1: std::rotate(data_.begin(), data_.begin() + 1, data_.end()); break;
                    case 2: for (auto& v : data_) v = std::fmod(v + 1.0, 10.0); break;
                    case 3:
                        for (std::size_t i = 0; i < data_.size() / 2; ++i) {
                            std::swap(data_[i], data_[data_.size() - 1 - i]);
                        }
                        break;
                    default: break;
                }
                return {{uik::Tensor({64}, data_)}, -0.01, step_ >= 100};
            }
            int action_space_size() const override { return 4; }
        private:
            std::vector<uik::Real> data_;
            std::size_t step_ = 0;
        };

        uik::agent::AgentKernel::Config cfg;
        cfg.wm_config.input_dim    = 64;
        cfg.wm_config.latent_dim   = 8;
        cfg.wm_config.action_space = 4;
        cfg.planning_horizon       = 3;
        cfg.evolve_interval        = 8;
        cfg.induce_interval        = 12;
        cfg.planner_config.action_space = 4;

        uik::agent::AgentKernel kernel(cfg);
        SimpleEnv env;
        kernel.run(env, 100);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Steps:         " << kernel.step_count() << std::endl;
        std::cout << "  Total reward:  " << kernel.total_reward() << std::endl;
        std::cout << "  Rules induced: " << kernel.rule_library().size() << std::endl;
        std::cout << "  Archive size:  " << kernel.evolution().archive().size()
                  << std::endl;
    }

    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "Done. Kernel operational." << std::endl;
    return 0;
}
