#include <omp.h>
#include <cmath>
#include <iostream>
#include <chrono>

using namespace std;
double result_time1[81];
double result_speed1[81];
double result_time2[81];
double result_speed2[81];

int number_threads = 0;

double completion_criteria1(double* a, double* b, double* x, int n) {
    double sum = 0.0, sum_b = 0.0;
    #pragma omp parallel for num_threads(number_threads)
        for(int i = 0; i < n; i++) {
            double current = 0, current_b = 0;
            for(int j = 0; j < n; j++)
                current += a[i * n + j] * x[j];
            current -= b[i];
            current = pow(current, 2);
            current_b = pow(b[i], 2);
            #pragma omp atomic
            sum += current;
            #pragma omp atomic
            sum_b += current_b;
        }
    return sqrt(sum)/ sqrt(sum_b);
}

double completion_criteria2(double* a, double* b, double* x, int n) {
    double sum = 0.0, sum_b = 0.0;
    #pragma omp for
        for(int i = 0; i < n; i++) {
            double current = 0, current_b = 0;
            for(int j = 0; j < n; j++)
                current += a[i * n + j] * x[j];
            current -= b[i];
            current = pow(current, 2);
            current_b = pow(b[i], 2);
            #pragma omp atomic
            sum += current;
            #pragma omp atomic
            sum_b += current_b;
    }
    return sqrt(sum)/ sqrt(sum_b);
}

void solving_system_linear_equations_1(double* a, double* b, double* x, int n) {
    double* x_current = (double*)malloc(sizeof(double) * n);
    while(completion_criteria1(a, b, x, n) > 0.00001){
        #pragma omp parallel for num_threads(number_threads)
            for(int i = 0; i < n; i++) {
                double current = 0;
                x_current[i] = 0;
                for(int j = 0; j < n; j++)
                    current += a[i * n + j] * x[j];
                x_current[i] = 0.0001*(current - b[i]);
            }
        #pragma omp parallel for num_threads(number_threads)
            for(int i = 0; i < n; i++)
                x[i] -= x_current[i];
    }
}

void solving_system_linear_equations_2(double* a, double* b, double* x, int n) {
        double* x_current = (double*)malloc(sizeof(double) * n);
        #pragma omp parallel num_threads(number_threads)
        {
            while(completion_criteria2(a, b, x, n) > 0.00001){
                #pragma omp for
                    for(int i = 0; i < n; i++) {
                        x_current[i] = 0;
                        double current = 0;
                        for(int j = 0; j < n; j++)
                            current += a[i * n + j] * x[j];
                        x_current[i] = 0.0001*(current - b[i]);
                    }
                #pragma omp for
                    for(int i = 0; i < n; i++)
                        x[i] -= x_current[i];
            }
        }
}

void run_parallel(size_t n, void (*func)(double*, double*, double*, int), int numb) {
    double *a, *b, *x;

    a = (double*)malloc(sizeof(*a) * n * n);
    b = (double*)malloc(sizeof(*b) * n);
    x = (double*)malloc(sizeof(*x) * n);

    if(a == NULL || b == NULL || x == NULL) {
        free(a);
        free(b);
        free(x);
        cout << "Error allocate memory!" << endl;
        exit(1);
    }

    #pragma omp parallel for num_threads(number_threads)
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++)
                a[i * n + j] = (i == j) ? 2 : 1;
            b[i] = n + 1;
            x[i] = 0.0;
        }

    const auto start{chrono::steady_clock::now()};
    func(a, b, x, n);
    const auto end{chrono::steady_clock::now()};
    const chrono::duration<double> elapsed_seconds{end - start};

    if(numb == 1){
        result_time1[number_threads] = elapsed_seconds.count();
        result_speed1[number_threads] =  result_time1[1] / result_time1[number_threads];
    }
    else{
        result_time2[number_threads] = elapsed_seconds.count();
        result_speed2[number_threads] =  result_time2[1] / result_time2[number_threads];
    }
    cout << "Elapsed time:" << elapsed_seconds.count() << "sec" << endl;
    
    free(a);
    free(b);
    free(x);
}

int main(int argc, char *argv[]) {
    int n = 16000;
    for (int i = 1; i <= 80; i++){
        number_threads = i;
        cout << i << " thread" << endl;
        cout << "The first program is running!" << endl;
        run_parallel(n, solving_system_linear_equations_1, 1);
        cout << "The second program is running!"<< endl;
        run_parallel(n, solving_system_linear_equations_2, 2);
        cout << endl;
    }
    for(int i = 1; i <= 80; i++)
        cout << result_time1[i] << " ";
    cout << endl;
    for(int i = 1; i <= 80; i++)
        cout << result_speed1[i] << " ";
    cout << endl;
    for(int i = 1; i <= 80; i++)
        cout << result_time2[i] << " ";
    cout << endl;
    for(int i = 1; i <= 80; i++)
        cout << result_speed2[i] << " ";
    cout << endl;
    return 0;
}