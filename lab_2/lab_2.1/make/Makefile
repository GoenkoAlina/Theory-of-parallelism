TORGET = lab_2.1
CXX = g++

$(TORGET) : lab_2.1.o
	$(CXX) -fopenmp $^ -o $@

lab_2.1.o : lab_2.1.c
	$(CXX) -c $^ -o $@ -fopenmp
