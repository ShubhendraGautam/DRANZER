CC = clang
AR = ar
# -O2 (not -O0/-g by default): the tokenizer/hashmap library was previously
# built unoptimized, which hurts on slow/low-end CPUs. No -march here, so
# the archive stays portable across machines.
CFLAGS = -I include/ -O2 -MMD -MP

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

# Only compile the core utility libraries
SRC = src/hashmap.c src/byte_pair_encoding.c
OBJ = $(SRC:.c=.o)
DEPS = $(OBJ:.o=.d)

TARGET = libattention.a

all: $(TARGET)

$(TARGET): $(OBJ)
	$(AR) rcs $(TARGET) $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(DEPS) $(TARGET)

-include $(DEPS)

.PHONY: all clean
