CXX := g++

ifeq ($(debug),1)
	DEBUG := -DDEBUG
	OPTS := -O0 -g
else
	OPTS := -Ofast -fno-unroll-loops -Os
endif

WARN     := -Wall -Wextra
CXXFLAGS := $(WARN) $(OPTS) $(DEBUG) -std=c++23 -I/usr/include
LIBS     := -lsfml-graphics -lsfml-window -lsfml-system

SRC_DIR     := src
INCLUDE_DIR := include
BUILD_DIR   := build

TARGET := $(BUILD_DIR)/nn

SRC := $(shell find $(SRC_DIR) -name '*.cpp')
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.cpp.o,$(SRC))
DIR := $(sort $(dir $(OBJ)))

RED    := \033[91m
YELLOW := \033[93m
GREEN  := \033[92m
BLUE   := \033[94m
RESET  := \033[0m

all: $(TARGET)

$(TARGET): $(OBJ)
	@printf "$(BLUE)  LD     Linking $@\n$(RESET)"
	@$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(LIBS)
ifeq ($(debug),1)
	@printf "$(YELLOW)  WARN   Warning: Compiling in DEBUG MODE\n"
endif

$(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp | $(DIR)
	@printf "$(GREEN)  CXX    Building object $@\n$(RESET)"
	@$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(GPU) -c -o $@ $<

$(DIR):
	@mkdir -p $(DIR)

clean:
	@printf "$(RED)  RM     Building directory $(BUILD_DIR)/\n$(RESET)"
	@rm -rf $(BUILD_DIR)

run:
	@printf "$(YELLOW)  RUN    Running executable $(TARGET)\n$(RESET)"
ifeq ($(debug),1)
	@gdb $(TARGET)
else
	@./$(TARGET)
endif
	@printf "$(YELLOW)  RUN    Done running executable $(TARGET)\n$(RESET)"

size:
	@wc -c < $(TARGET) | awk '{printf "%.2f KB\n", $$1 / 1000}'

decompress_datasets: requirements
	@[ -f datasets/mnist/mnist_train.csv ] || unzip datasets/mnist/mnist_train.csv.zip -d datasets/mnist/
	@[ -f datasets/mnist/mnist_test.csv ]  || unzip datasets/mnist/mnist_test.csv.zip  -d datasets/mnist/
	@[ -f datasets/fashion/train-images-idx3-ubyte ] || gzip -dk datasets/fashion/train-images-idx3-ubyte.gz
	@[ -f datasets/fashion/train-labels-idx1-ubyte ]  || gzip -dk datasets/fashion/train-labels-idx1-ubyte.gz
	@python3 ubyte-to-csv.py

clean-datasets:
	@rm -rf datasets/mnist/mnist_train.csv
	@rm -rf datasets/mnist/mnist_test.csv
	@rm -rf datasets/fashion/train-images-idx3-ubyte
	@rm -rf datasets/fashion/train-labels-idx1-ubyte
	@rm -rf datasets/fashion/fashion_mnist_train.csv

requirements:
	@pip install numpy
