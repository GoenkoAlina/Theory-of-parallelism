TARGET = lab_7
CXX = pgc++

CFLAGS = -I/home/a.goenko/lab_6/boost_1_85_0/boost -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include -fast -ta=tesla -acc=gpu -Minfo=all

FLAGS = -L/home/a.goenko/lab_6/boost_1_85_0/boost/libs -lstdc++ -lboost_program_options -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include -cudalib=cublas

$(TARGET) : lab_7.o
	$(CXX) $(CFLAGS) -o $@ $^ $(FLAGS)

lab_7.o : main.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^ $(FLAGS)
