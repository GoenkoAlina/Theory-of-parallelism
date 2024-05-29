TARGET = lab_6
CXX = pgc++

$(TARGET) : lab_6.o
	$(CXX) $^ -L /home/a.goenko/lab_6/boost_1_85_0/boost/libs -lstdc++ -lboost_program_options -fast -ta=tesla -acc=gpu -Minfo=all -o $@ -std=c++20

lab_6.o : main.cpp
	$(CXX) -c $^ -I /home/a.goenko/lab_6/boost_1_85_0/boost -fast -ta=tesla -acc=gpu -Minfo=all -o $@ -std=c++20
