TORGET = lab_2.2
CXX = g++

$(TORGET): lab_2.2.o
	$(CXX) -fopenmp $^ -o $@

lab_2.2.o : lab_2.2.c
	$(CXX) -c $^ -o $@ -fopenmp