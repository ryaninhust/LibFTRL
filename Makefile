CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: train predict

train: train.cpp ftrl.o
	$(CXX) $(CXXFLAGS) -o $@ $^

predict: predict.cpp ftrl.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: ftrl.cpp ftrl.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f train predict ftrl.o *.bin.*
