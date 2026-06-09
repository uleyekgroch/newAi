#include "agent/arc_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace uik::agent {

namespace {

// Minimal JSON parser for ARC format (no external dependency)
// Handles: {"train": [...], "test": [...]} where each element has "input" and "output" as 2D int arrays

void skip_whitespace(const std::string& s, std::size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
}

void expect_char(const std::string& s, std::size_t& pos, char c) {
    skip_whitespace(s, pos);
    if (pos >= s.size() || s[pos] != c) {
        throw std::runtime_error(std::string("Expected '") + c + "' at position " + std::to_string(pos));
    }
    ++pos;
}

std::string parse_string(const std::string& s, std::size_t& pos) {
    skip_whitespace(s, pos);
    expect_char(s, pos, '"');
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        result += s[pos++];
    }
    expect_char(s, pos, '"');
    return result;
}

int parse_int(const std::string& s, std::size_t& pos) {
    skip_whitespace(s, pos);
    std::size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
    if (pos == start) throw std::runtime_error("Expected integer at position " + std::to_string(pos));
    return std::stoi(s.substr(start, pos - start));
}

std::vector<int> parse_int_array(const std::string& s, std::size_t& pos) {
    std::vector<int> result;
    expect_char(s, pos, '[');
    skip_whitespace(s, pos);
    if (pos < s.size() && s[pos] == ']') { ++pos; return result; }
    result.push_back(parse_int(s, pos));
    while (true) {
        skip_whitespace(s, pos);
        if (pos >= s.size() || s[pos] == ']') break;
        expect_char(s, pos, ',');
        result.push_back(parse_int(s, pos));
    }
    expect_char(s, pos, ']');
    return result;
}

std::vector<std::vector<int>> parse_2d_int_array(const std::string& s, std::size_t& pos) {
    std::vector<std::vector<int>> result;
    expect_char(s, pos, '[');
    skip_whitespace(s, pos);
    if (pos < s.size() && s[pos] == ']') { ++pos; return result; }
    result.push_back(parse_int_array(s, pos));
    while (true) {
        skip_whitespace(s, pos);
        if (pos >= s.size() || s[pos] == ']') break;
        expect_char(s, pos, ',');
        result.push_back(parse_int_array(s, pos));
    }
    expect_char(s, pos, ']');
    return result;
}

ArcTask::Pair parse_pair(const std::string& s, std::size_t& pos) {
    ArcTask::Pair pair;
    expect_char(s, pos, '{');
    while (true) {
        skip_whitespace(s, pos);
        if (pos >= s.size() || s[pos] == '}') break;
        if (s[pos] == ',') { ++pos; continue; }
        std::string key = parse_string(s, pos);
        expect_char(s, pos, ':');
        if (key == "input") {
            pair.input = parse_2d_int_array(s, pos);
        } else if (key == "output") {
            pair.output = parse_2d_int_array(s, pos);
        }
    }
    expect_char(s, pos, '}');
    return pair;
}

std::vector<ArcTask::Pair> parse_pair_array(const std::string& s, std::size_t& pos) {
    std::vector<ArcTask::Pair> result;
    expect_char(s, pos, '[');
    skip_whitespace(s, pos);
    if (pos < s.size() && s[pos] == ']') { ++pos; return result; }
    result.push_back(parse_pair(s, pos));
    while (true) {
        skip_whitespace(s, pos);
        if (pos >= s.size() || s[pos] == ']') break;
        expect_char(s, pos, ',');
        result.push_back(parse_pair(s, pos));
    }
    expect_char(s, pos, ']');
    return result;
}

} // anonymous namespace

ArcTask ArcLoader::load_task(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open ARC task file: " + path.string());
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();

    ArcTask task;
    task.id = path.stem().string();

    std::size_t pos = 0;
    expect_char(content, pos, '{');
    while (true) {
        skip_whitespace(content, pos);
        if (pos >= content.size() || content[pos] == '}') break;
        if (content[pos] == ',') { ++pos; continue; }
        std::string key = parse_string(content, pos);
        expect_char(content, pos, ':');
        if (key == "train") {
            task.train = parse_pair_array(content, pos);
        } else if (key == "test") {
            task.test = parse_pair_array(content, pos);
        }
    }

    return task;
}

std::vector<ArcTask> ArcLoader::load_directory(const std::filesystem::path& dir,
                                                 std::size_t max_tasks) {
    std::vector<ArcTask> tasks;
    std::vector<std::filesystem::path> paths;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".json") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());

    if (max_tasks > 0 && paths.size() > max_tasks) {
        paths.resize(max_tasks);
    }

    for (const auto& path : paths) {
        try {
            tasks.push_back(load_task(path));
        } catch (...) {
            // Skip malformed files
        }
    }
    return tasks;
}

Tensor ArcLoader::grid_to_tensor(const std::vector<std::vector<int>>& grid) {
    if (grid.empty()) return Tensor({0}, 0.0);
    Dim rows = grid.size();
    Dim cols = grid[0].size();
    std::vector<Real> data;
    data.reserve(rows * cols);
    for (const auto& row : grid) {
        for (int val : row) {
            data.push_back(static_cast<Real>(val));
        }
    }
    return Tensor({rows * cols}, std::move(data));
}

std::vector<std::vector<int>> ArcLoader::tensor_to_grid(const Tensor& t,
                                                          Dim rows, Dim cols) {
    std::vector<std::vector<int>> grid(rows, std::vector<int>(cols, 0));
    for (Dim r = 0; r < rows; ++r) {
        for (Dim c = 0; c < cols; ++c) {
            Dim idx = r * cols + c;
            if (idx < t.flat_size()) {
                grid[r][c] = static_cast<int>(std::round(t.at(idx)));
            }
        }
    }
    return grid;
}

bool ArcLoader::check_exact_match(const std::vector<std::vector<int>>& output,
                                    const std::vector<std::vector<int>>& expected) {
    if (output.size() != expected.size()) return false;
    for (std::size_t r = 0; r < output.size(); ++r) {
        if (output[r].size() != expected[r].size()) return false;
        for (std::size_t c = 0; c < output[r].size(); ++c) {
            if (output[r][c] != expected[r][c]) return false;
        }
    }
    return true;
}

double ArcLoader::cell_accuracy(const std::vector<std::vector<int>>& output,
                                  const std::vector<std::vector<int>>& expected) {
    if (expected.empty()) return 0.0;
    std::size_t total = 0;
    std::size_t correct = 0;
    Dim rows = expected.size();
    for (Dim r = 0; r < rows; ++r) {
        Dim cols = expected[r].size();
        for (Dim c = 0; c < cols; ++c) {
            ++total;
            if (r < output.size() && c < output[r].size() &&
                output[r][c] == expected[r][c]) {
                ++correct;
            }
        }
    }
    return total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
}

} // namespace uik::agent
