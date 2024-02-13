TORGET = lab_1
CXX = g++

$(TORGET): lab_1.o
	$(CXX) $^ -o $@

lab_1.o : lab_1.cpp
	$(CXX) -c $^ -o $@

double.o : lab_1.cpp
	$(CXX) -D TYPE -c $^ -o $@

float: lab_1.o
	$(CXX) $^ -o $(TORGET)

double: double.o
	$(CXX) $^ -o $(TORGET)

