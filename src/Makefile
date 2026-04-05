CC = clang
CFLAGS = -I include/ -I ../libs/include/ -g
LDFLAGS = -lm

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

# Main source - unified neural model implementation
MAIN_SOURCE = main.c

# Source files for src directory
SRC = $(MAIN_SOURCE) tokenizer.c attention.c model.c config.c sampling.c batch.c checkpoint.c cli.c
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
