#include <iostream>
#include <cstdlib>
#include <chrono>
#include <program_options.hpp>

int size;
double eps;
int n_iter;

void linear_interpolation(double* A, double* newA, int from, int to, int step){
    double difference;
    difference = 10.0 / (size - 1);
    for(int i = from; i < to; i+=step){
        A[i] = newA[i] = A[i - step] + difference;
    }
}

void init(double* A, double* newA){
        A[0] = newA[0] = 10.0;
        A[size - 1] = newA[size - 1] = 20.0;
        A[size * (size - 1)] = newA[size * (size - 1)] = 20.0;
        A[size * size - 1] = newA[size  * size - 1] = 30.0;
        linear_interpolation(A, newA, 1, size - 1, 1);
        linear_interpolation(A, newA, size * (size - 1) + 1, size * size - 1, 1);
        linear_interpolation(A, newA, size, size * (size - 1), size);
        linear_interpolation(A, newA, size * 2 - 1, size * size - 1, size);
}
//
void solve(double* A, double* newA, double* ptr){
    double error = 1.0;
    int iter = 0;

#pragma acc data copy(A[0:size*size], newA[0:size*size], ptr[0:1])
    while ((error > eps) && (iter < n_iter)) {
        iter++;
        error = 0.0;

#pragma acc parallel loop reduction(max:error) present(A, newA) async
        for (int i = 1; i < size - 1; i++) {

#pragma acc loop
            for (int j = 1; j < size - 1; j++) {
                newA[size * i + j] = 0.25 * (A[size * i + j - 1] + A[size * i + j + 1] + A[size * (i - 1) + j] + A[size * (i + 1) + j]);
                error = fmax(error, fabs(A[size * i + j] - newA[size * i + j]));
            }
        }
#pragma acc wait

        ptr = A;
        A = newA;
        newA = ptr;
    }
    std::cout << iter << ": " << error << std::endl;
}

int main(int argc, char** argv){
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
             ("accuracy", boost::program_options::value<double>(&eps), "accuracy")
             ("grid_size", boost::program_options::value<int>(&size), "grid size")
             ("number_iter", boost::program_options::value<int>(&n_iter), "number of iterations");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    auto* A = (double*)calloc(size * size, sizeof(double));
    auto* newA = (double*)calloc(size * size, sizeof(double));
    double* ptr = (double*)calloc(1, sizeof(double));

    init(A, newA);

    const auto start{std::chrono::steady_clock::now()};

    solve(A, newA, ptr);

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    std::cout << "Time: " << elapsed_seconds.count() << std::endl;

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            std::cout << A[i * size + j] << " ";
        }
        std::cout << std:: endl;
    }

    free(A);
    free(newA);

    return 0;
}
