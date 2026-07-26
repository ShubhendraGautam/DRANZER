CC = clang
# -ffast-math is portable (no special CPU features required) and helps on
# weak FPUs. Deliberately NOT using -march=native by default: a binary
# built with it only runs on the exact CPU that compiled it and raises
# SIGILL ("illegal instruction") on any other machine - including most
# low-end/older hardware you'd want to deploy this to.
CFLAGS = -I include/ -I ../libs/include/ -O3 -ffast-math
LDFLAGS = -lm

# Tune for the exact CPU you're building on: make NATIVE=1
# Fastest option, but the resulting binary is NOT portable to other machines.
ifdef NATIVE
    CFLAGS += -march=native
    $(info Native CPU tuning enabled (NATIVE=1) - binary will only run on this build machine's CPU)
endif

# Optimize for code size instead of speed: make SIZE=1
# Useful on storage- or icache-constrained low-end/embedded devices.
ifdef SIZE
    CFLAGS := $(filter-out -O3,$(CFLAGS)) -Os
    $(info Size-optimized build enabled (SIZE=1))
endif

# Enable debug output with: make DEBUG=1
ifdef DEBUG
    CFLAGS += -g -DDEBUG
    $(info DEBUG mode enabled)
endif

# Enable AddressSanitizer with: make ASAN=1
ifdef ASAN
    CFLAGS += -fsanitize=address -fno-omit-frame-pointer -O1 -g
    $(info AddressSanitizer enabled)
endif

# Main source - unified neural model implementation
MAIN_SOURCE = main.c

# Source files for src directory
SRC = $(MAIN_SOURCE) tokenizer.c attention.c model.c config.c sampling.c batch.c checkpoint.c cli.c stream.c
OBJ = $(SRC:.c=.o)

TARGET = app.out

# Library
LIB_DIR = ../libs
LIB_NAME = attention
LIB = $(LIB_DIR)/lib$(LIB_NAME).a

all: $(LIB) $(TARGET)

# Step 1: Build library first
$(LIB):
	$(MAKE) DEBUG=$(DEBUG) ASAN=$(ASAN) -C $(LIB_DIR)

# Step 2: Link everything
$(TARGET): $(OBJ) $(LIB)
	$(CC) $(OBJ) -L$(LIB_DIR) -l$(LIB_NAME) $(CFLAGS) $(LDFLAGS) -o $(TARGET)

# Step 3: Compile src files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
	$(MAKE) -C $(LIB_DIR) clean

.PHONY: all clean
