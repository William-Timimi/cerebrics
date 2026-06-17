CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2 -g
LINT_FLAGS := -std=c11 -Wall -Wextra -Wpedantic -Iinclude
CPPFLAGS ?= -Iinclude -Itests
LDFLAGS ?=
LDLIBS ?= -lm
PYTHON ?= python
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy

BUILD_DIR := build
CORE_SRCS := src/cerebrics.c src/errors.c
PUBLIC_HEADERS := include/cerebrics/cerebrics.h include/cerebrics/errors.h
TEST_UTIL_SRCS := tests/test_utils.c
TEST_SRCS := $(filter-out $(TEST_UTIL_SRCS),$(wildcard tests/test_*.c))
TEST_BINS := $(patsubst tests/%.c,$(BUILD_DIR)/%,$(TEST_SRCS))
EXAMPLE ?= tensor_basics
EXAMPLE_SRC := examples/$(EXAMPLE).c
TARGET := $(BUILD_DIR)/example_$(EXAMPLE)
C_SOURCES := $(CORE_SRCS) $(wildcard examples/*.c) $(TEST_SRCS) $(TEST_UTIL_SRCS)

.PHONY: all run example examples list-examples test test-c format lint check clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(CORE_SRCS) $(EXAMPLE_SRC) $(PUBLIC_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(CORE_SRCS) $(EXAMPLE_SRC) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/test_%: tests/test_%.c $(TEST_UTIL_SRCS) $(CORE_SRCS) $(PUBLIC_HEADERS) tests/test_utils.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(CORE_SRCS) $(TEST_UTIL_SRCS) $< -o $@ $(LDFLAGS) $(LDLIBS)

run example: $(TARGET)
	./$(TARGET)

examples: $(BUILD_DIR)
	@for src in examples/*.c; do \
		name=$$(basename $$src .c); \
		echo "==> building $$name"; \
		$(CC) $(CPPFLAGS) $(CFLAGS) $(CORE_SRCS) $$src -o $(BUILD_DIR)/example_$$name $(LDFLAGS) $(LDLIBS); \
	done

list-examples:
	@for src in examples/*.c; do basename $$src .c; done

test: test-c

test-c: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "==> $$t"; ./$$t; done

FORMAT_FILES := include/cerebrics/*.h src/*.c tests/*.c examples/*.c
TIDY_FILES := src/*.c tests/*.c examples/*.c

format:
	clang-format -i $(FORMAT_FILES)

lint:
	clang-tidy $(TIDY_FILES) -- $(LINT_FLAGS)

lint-fix:
	clang-tidy $(TIDY_FILES) --fix -- $(LINT_FLAGS)
	clang-format -i $(FORMAT_FILES)

lint-fix-errors:
	clang-tidy $(TIDY_FILES) --fix --fix-errors -- $(LINT_FLAGS)
	clang-format -i $(FORMAT_FILES)

check: format lint test-c

clean:
	rm -rf $(BUILD_DIR)