#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

using namespace std;

double par;

void init(double *a, double *b, double *c, int m, int n, int nthreads, int threadid){
    int items_per_thread = m / nthreads;
    int lb = threadid * items_per_thread;
    int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
    for (int i = lb; i <= ub; i++) {
        for (int j = 0; j < n; j++){
            a[i * n + j] = i + j;
        }
        c[i] = 0.0;
    }
    items_per_thread = n / nthreads;
    lb = threadid * items_per_thread;
    ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
    for (int i = lb; i <= ub; i++)
        b[i] = i;
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_parallel(double *a, double *b, double *c, int m, int n, int nthreads, int threadid)
{
    int items_per_thread = m / nthreads;
    int lb = threadid * items_per_thread;
    int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
    for (int i = lb; i <= ub; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void run_serial(size_t n, size_t m)
{
    double *a, *b, *c;
    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        cout << "Error allocate memory!" << endl;
        exit(1);
    }

    const auto start{chrono::steady_clock::now()};

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    matrix_vector_product(a, b, c, m, n);
    const auto end{chrono::steady_clock::now()};
    const chrono::duration<double> elapsed_seconds{end - start};
    par = elapsed_seconds.count();

    cout << "Elapsed time (serial): " << par << " sec." << endl;
    free(a);
    free(b);
    free(c);
}

void run_parallel(size_t n, size_t m, int nthreads) {
    double *a, *b, *c;
    vector<jthread> threads;

    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
        {
            free(a);
            free(b);
            free(c);
            printf("Error allocate memory!\n");
            exit(1);
        }
    const auto start{chrono::steady_clock::now()};
    for(int threadid = 0; threadid < nthreads; threadid++){
        threads.emplace_back(init, a, b, c, m, n, nthreads, threadid);
    }
    threads.clear();

    for(int threadid = 0; threadid < nthreads; threadid++){
        threads.emplace_back(matrix_vector_product_parallel, a, b, c, m, n, nthreads, threadid);
    }
    threads.clear();
    const auto end{chrono::steady_clock::now()};
    const chrono::duration<double> elapsed_seconds{end - start};

    cout << "Elapsed time (parallel): " << elapsed_seconds.count() << " sec." << endl;
    cout << "Speed: " << par/elapsed_seconds.count() << endl;
    free(a);
    free(b);
    free(c);
}

int main(int argc, char *argv[])
{
    size_t M = 20000;
    size_t N = 20000;
    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        N = atoi(argv[2]);
    int threads_count[8] = {1, 2, 4, 7, 8, 16, 20, 40};
    for(int i = 0; i < 8; i++){
        cout << threads_count[i] << endl;
        run_serial(M, N);
        run_parallel(M, N, threads_count[i]);
    }
    return 0;
}