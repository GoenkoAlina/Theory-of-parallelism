TORGET = lab_2.3
CXX = g++

$(TORGET): lab_2.3.o
	$(CXX) -fopenmp $^ -o $@

lab_2.3.o : lab_2.3.cpp
	$(CXX) -c $^ -o $@ -fopenmp