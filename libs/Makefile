CC = clang
AR = ar
CFLAGS = -I include/ -g

# Enable debug output with: make DEBUG=1
ifdef DEBUG
    CFLAGS += -DDEBUG
    $(info DEBUG mode enabled)
endif

# Enable AddressSanitizer with: make ASAN=1
ifdef ASAN
    CFLAGS += -fsanitize=address -fno-omit-frame-pointer -O1
    $(info AddressSanitizer enabled)
endif

# Only compile the core utility libraries
SRC = src/hashmap.c src/byte_pair_encoding.c
OBJ = $(SRC:.c=.o)

TARGET = libattention.a

all: $(TARGET)

$(TARGET): $(OBJ)
	$(AR) rcs $(TARGET) $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean