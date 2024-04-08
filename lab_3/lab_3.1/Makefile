TARGET = lab_3.1
CXX = g++
version = -std=c++20

$(TARGET): lab_3.1.o
	$(CXX) $^ $(version) -o $@

lab_3.1.o : lab_3.1.cpp
	$(CXX) -c $^ $(version) -o $@
