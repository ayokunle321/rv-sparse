CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
INCLUDES = -Iinclude

SRC = \
	src/core/matrix.c \
	src/core/error.c \
	src/core/spgemm.c \
	src/kernels/spgemm/csr_scalar_f32.c

EXAMPLE = examples/spgemm_csr_f32.c
BUILD_DIR = build
TARGET = $(BUILD_DIR)/spgemm_csr_f32

.PHONY: all run clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) $(EXAMPLE) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) $(EXAMPLE) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
