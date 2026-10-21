CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic

HEADERS := $(wildcard src/*.hpp)

all: build/fxprice

build/fxprice: src/main.cpp $(HEADERS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/main.cpp -o $@

build/test_core: tests/test_core.cpp $(HEADERS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_core.cpp -o $@

# Run the experiment and write dashboard/data.js + dashboard/results.json
run: build/fxprice
	./build/fxprice --seeds 10 --out dashboard

test: build/test_core
	./build/test_core

clean:
	rm -rf build

.PHONY: all run test clean
