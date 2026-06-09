#include "agent/planner.hpp"
#include <limits>
#include <algorithm>
#include <cmath>

namespace uik::agent {

Planner::Planner() : Planner(Config{}) {}

Planner::Planner(Config config) : config_(config), rng_(config.seed) {}

std::vector<Action> Planner::plan(const State& current,
                                    const State& goal,
                                    IWorldModel& world_model,
                                    int horizon) {
    if (horizon <= 0) return {};

    if (config_.use_mcts) {
        return mcts_plan(current, goal, world_model, horizon);
    }

    // Fallback: random-shooting
    std::vector<Action> best_plan;
    Real best_dist = std::numeric_limits<Real>::max();

    std::uniform_int_distribution<int> action_dist(0, config_.action_space - 1);

    for (std::size_t sample = 0; sample < config_.num_samples; ++sample) {
        std::vector<Action> candidate_plan;
        candidate_plan.reserve(static_cast<std::size_t>(horizon));
        State sim_state = current;

        for (int t = 0; t < horizon; ++t) {
            Action a{action_dist(rng_)};
            candidate_plan.push_back(a);
            try {
                sim_state = world_model.predict_next(sim_state, a);
            } catch (...) {
                break;
            }
        }

        Real dist = state_distance(sim_state, goal);
        if (dist < best_dist) {
            best_dist = dist;
            best_plan = std::move(candidate_plan);
        }
    }

    return best_plan;
}

// MCTS: Monte Carlo Tree Search with UCB1 selection
std::vector<Action> Planner::mcts_plan(const State& current,
                                         const State& goal,
                                         IWorldModel& world_model,
                                         int horizon) {
    auto root = std::make_unique<MCTSNode>();
    root->expanded = false;

    for (std::size_t sim = 0; sim < config_.mcts_simulations; ++sim) {
        mcts_simulate(root.get(), current, goal, world_model, 0, horizon);
    }

    // Select best action: child with highest average value
    if (root->children.empty()) {
        // Fallback if MCTS didn't expand
        return {Action{0}};
    }

    MCTSNode* best_child = nullptr;
    Real best_value = -std::numeric_limits<Real>::max();
    for (auto& child : root->children) {
        if (child->visits > 0) {
            Real avg = child->total_value / static_cast<Real>(child->visits);
            if (avg > best_value) {
                best_value = avg;
                best_child = child.get();
            }
        }
    }

    if (!best_child) return {Action{0}};

    // Build plan: follow best path down the tree
    std::vector<Action> plan;
    MCTSNode* node = best_child;
    while (node != nullptr && static_cast<int>(plan.size()) < horizon) {
        plan.push_back(node->action);
        // Find best child at each level
        MCTSNode* next_best = nullptr;
        Real next_best_val = -std::numeric_limits<Real>::max();
        for (auto& child : node->children) {
            if (child->visits > 0) {
                Real avg = child->total_value / static_cast<Real>(child->visits);
                if (avg > next_best_val) {
                    next_best_val = avg;
                    next_best = child.get();
                }
            }
        }
        node = next_best;
    }

    // Pad plan to full horizon if tree didn't extend deep enough
    std::uniform_int_distribution<int> action_dist(0, config_.action_space - 1);
    while (static_cast<int>(plan.size()) < horizon) {
        plan.push_back(Action{action_dist(rng_)});
    }

    return plan;
}

Real Planner::mcts_simulate(MCTSNode* node, const State& state,
                              const State& goal, IWorldModel& world_model,
                              int depth, int max_depth) {
    // Terminal: evaluate state distance to goal (negative = better)
    if (depth >= max_depth) {
        Real dist = state_distance(state, goal);
        Real value = -dist;  // closer to goal = higher value
        node->visits++;
        node->total_value += value;
        return value;
    }

    // Expand if not yet expanded
    if (!node->expanded) {
        mcts_expand(node);
        // Random rollout from this node
        std::uniform_int_distribution<int> action_dist(0, config_.action_space - 1);
        State sim_state = state;
        for (int t = depth; t < max_depth; ++t) {
            Action a{action_dist(rng_)};
            try {
                sim_state = world_model.predict_next(sim_state, a);
            } catch (...) {
                break;
            }
        }
        Real dist = state_distance(sim_state, goal);
        Real value = -dist;
        node->visits++;
        node->total_value += value;
        return value;
    }

    // Select child via UCB1
    MCTSNode* selected = mcts_select(node);
    if (!selected) return 0.0;

    // Simulate the action
    State next_state = state;
    try {
        next_state = world_model.predict_next(state, selected->action);
    } catch (...) {
        // If prediction fails, penalize
        selected->visits++;
        selected->total_value -= 10.0;
        node->visits++;
        node->total_value -= 10.0;
        return -10.0;
    }

    // Recurse
    Real value = mcts_simulate(selected, next_state, goal, world_model,
                                depth + 1, max_depth);
    node->visits++;
    node->total_value += value;
    return value;
}

Planner::MCTSNode* Planner::mcts_select(MCTSNode* node) const {
    if (node->children.empty()) return nullptr;

    MCTSNode* best = nullptr;
    Real best_ucb = -std::numeric_limits<Real>::max();

    for (auto& child : node->children) {
        Real u = ucb1(child.get());
        if (u > best_ucb) {
            best_ucb = u;
            best = child.get();
        }
    }
    return best;
}

void Planner::mcts_expand(MCTSNode* node) {
    node->expanded = true;
    for (int a = 0; a < config_.action_space; ++a) {
        auto child = std::make_unique<MCTSNode>();
        child->action = Action{a};
        child->parent = node;
        node->children.push_back(std::move(child));
    }
}

Real Planner::ucb1(const MCTSNode* node) const {
    if (node->visits == 0) {
        return std::numeric_limits<Real>::max();  // unexplored → infinite priority
    }
    Real exploitation = node->total_value / static_cast<Real>(node->visits);
    Real parent_visits = node->parent ?
        static_cast<Real>(node->parent->visits) : 1.0;
    Real exploration = config_.exploration_constant *
        std::sqrt(std::log(parent_visits + 1.0) / static_cast<Real>(node->visits));
    return exploitation + exploration;
}

Real Planner::state_distance(const State& a, const State& b) const {
    if (a.latent.empty() || b.latent.empty()) return 0.0;
    Tensor diff = a.latent - b.latent;
    return diff.l2_norm();
}

} // namespace uik::agent
