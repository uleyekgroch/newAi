#pragma once

#include "common/types.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <functional>
#include <chrono>

namespace uik {

// Machine-readable structured logger.
// Outputs key=value pairs per AGENTS.md structured logging requirements.
class StructuredLogger {
public:
    enum class Level { Debug, Info, Warn, Error };

    struct Entry {
        Level level;
        std::string message;
        std::vector<std::pair<std::string, std::string>> fields;
        std::chrono::steady_clock::time_point timestamp;
    };

    using Sink = std::function<void(const Entry&)>;

    StructuredLogger() = default;

    void set_sink(Sink sink) { sink_ = std::move(sink); }
    void set_level(Level min_level) { min_level_ = min_level; }

    void log(Level level, const std::string& msg,
             std::vector<std::pair<std::string, std::string>> fields = {}) {
        if (level < min_level_) return;
        Entry entry{level, msg, std::move(fields),
                    std::chrono::steady_clock::now()};
        entries_.push_back(entry);
        if (sink_) sink_(entry);
    }

    void debug(const std::string& msg,
               std::vector<std::pair<std::string, std::string>> fields = {}) {
        log(Level::Debug, msg, std::move(fields));
    }

    void info(const std::string& msg,
              std::vector<std::pair<std::string, std::string>> fields = {}) {
        log(Level::Info, msg, std::move(fields));
    }

    void warn(const std::string& msg,
              std::vector<std::pair<std::string, std::string>> fields = {}) {
        log(Level::Warn, msg, std::move(fields));
    }

    void error(const std::string& msg,
               std::vector<std::pair<std::string, std::string>> fields = {}) {
        log(Level::Error, msg, std::move(fields));
    }

    [[nodiscard]] const std::vector<Entry>& entries() const { return entries_; }
    [[nodiscard]] std::size_t count() const { return entries_.size(); }

    // Format an entry as "level=INFO msg=... key1=val1 key2=val2"
    [[nodiscard]] static std::string format(const Entry& entry) {
        std::ostringstream oss;
        oss << "level=" << level_str(entry.level)
            << " msg=\"" << entry.message << "\"";
        for (const auto& [k, v] : entry.fields) {
            oss << " " << k << "=" << v;
        }
        return oss.str();
    }

    [[nodiscard]] static std::string level_str(Level l) {
        switch (l) {
            case Level::Debug: return "DEBUG";
            case Level::Info:  return "INFO";
            case Level::Warn:  return "WARN";
            case Level::Error: return "ERROR";
        }
        return "UNKNOWN";
    }

private:
    Sink sink_;
    Level min_level_ = Level::Info;
    std::vector<Entry> entries_;
};

} // namespace uik
