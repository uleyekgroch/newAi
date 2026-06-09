#include <gtest/gtest.h>
#include "common/logger.hpp"

using namespace uik;

TEST(StructuredLogger, initially_empty) {
    StructuredLogger logger;
    EXPECT_EQ(logger.count(), 0u);
}

TEST(StructuredLogger, info_records_entry) {
    StructuredLogger logger;
    logger.info("test_msg", {{"key", "val"}});
    EXPECT_EQ(logger.count(), 1u);
    EXPECT_EQ(logger.entries()[0].level, StructuredLogger::Level::Info);
    EXPECT_EQ(logger.entries()[0].message, "test_msg");
}

TEST(StructuredLogger, debug_filtered_by_default) {
    StructuredLogger logger;
    logger.debug("should_be_hidden");
    EXPECT_EQ(logger.count(), 0u);
}

TEST(StructuredLogger, debug_visible_when_level_set) {
    StructuredLogger logger;
    logger.set_level(StructuredLogger::Level::Debug);
    logger.debug("visible_now");
    EXPECT_EQ(logger.count(), 1u);
}

TEST(StructuredLogger, format_output) {
    StructuredLogger::Entry entry;
    entry.level = StructuredLogger::Level::Info;
    entry.message = "test";
    entry.fields = {{"step", "42"}, {"reward", "0.5"}};

    std::string formatted = StructuredLogger::format(entry);
    EXPECT_NE(formatted.find("level=INFO"), std::string::npos);
    EXPECT_NE(formatted.find("msg=\"test\""), std::string::npos);
    EXPECT_NE(formatted.find("step=42"), std::string::npos);
    EXPECT_NE(formatted.find("reward=0.5"), std::string::npos);
}

TEST(StructuredLogger, sink_receives_entries) {
    StructuredLogger logger;
    int sink_calls = 0;
    logger.set_sink([&sink_calls](const StructuredLogger::Entry&) {
        ++sink_calls;
    });

    logger.info("msg1");
    logger.warn("msg2");
    EXPECT_EQ(sink_calls, 2);
}

TEST(StructuredLogger, warn_and_error_always_recorded) {
    StructuredLogger logger;
    logger.set_level(StructuredLogger::Level::Warn);
    logger.info("filtered");
    logger.warn("visible");
    logger.error("also_visible");
    EXPECT_EQ(logger.count(), 2u);
}

TEST(StructuredLogger, level_str) {
    EXPECT_EQ(StructuredLogger::level_str(StructuredLogger::Level::Debug), "DEBUG");
    EXPECT_EQ(StructuredLogger::level_str(StructuredLogger::Level::Info), "INFO");
    EXPECT_EQ(StructuredLogger::level_str(StructuredLogger::Level::Warn), "WARN");
    EXPECT_EQ(StructuredLogger::level_str(StructuredLogger::Level::Error), "ERROR");
}
