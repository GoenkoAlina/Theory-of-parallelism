#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <cmath>
#include <functional>
#include <mutex>
#include <fstream>
#include <string>
#include <cassert>

std::mutex mut1;
std::mutex mut2;

std::queue<std::pair<size_t, std::packaged_task<double()>>> tasks;

std::unordered_map<size_t, double> results;

size_t id_task_;

std::condition_variable cv;

enum class task {sin, sqrt, pow};

template<typename T>
T fun_sin(T arg) {
    return std::sin(arg);
}

template<typename T>
T fun_sqrt(T arg) {
    return std::sqrt(arg);
}

template<typename T>
T fun_pow(T x, T y) {
    return std::pow(x, y);
}

template<typename T>
class Server {
public:
    void start(std::stop_token stoken){
        std::cout << "Start\n";
        size_t id_task;
        std::packaged_task<T()> task;
        // пока не получили сигнал стоп
        while (!stoken.stop_requested())
        {
            // если очередь не пуста, то достаем и решаем задчу
            if (!tasks.empty()) {
                {
                    std::lock_guard<std::mutex> lock_task{mut1};
                    id_task = tasks.front().first;
                    task = std::move(tasks.front().second);
                    tasks.pop();
                }
                auto future = task.get_future();
                task();
                {
                    std::lock_guard<std::mutex> lock_res{mut2};
                    results.insert({id_task, future.get()});
                }
                cv.notify_all();
            }
        }

        std::cout << "Server stop!\n";
    }

    void stop(std::jthread& server_thread){
        server_thread.request_stop();
    }

    size_t add_task(auto bind_){
        // создаем задачу (ленивое выполнение)
        std::packaged_task<T()> task(bind_);

        // добавляем задачу в очередь
        std::lock_guard<std::mutex> lock_task{mut1};
        ++id_task_;
        tasks.push({id_task_, std::move(task)});
        return id_task_;
    }

    T request_result(size_t id_task){
        T result;

        // блокировщик для работы с общими данными
        std::unique_lock<std::mutex> lock_res{mut2};
        cv.wait(lock_res, [id_task]{return results.find(id_task) != results.end();});
        result = results[id_task];
        results.erase(id_task);
        return result;
    }
};

template<typename T>
void client(Server<T> server, task t, int number_works){
    size_t id;
    T result;
    std::ofstream out;
    switch(t){
        case task::sin:
            out.open("sin.txt");
            for(int i = 0; i < number_works; i++){
                double arg = ((double)rand() / RAND_MAX) * 7;
                id = server.add_task(std::bind(fun_sin<T>, arg));
                result = server.request_result(id);
                out << arg << ' ' << result << '\n';
            }
            break;
        case task::sqrt:
            out.open("sqrt.txt");
            for(int i = 0; i < number_works; i++){
                double arg = ((double)rand() / RAND_MAX) * 100;
                id = server.add_task(std::bind(fun_sqrt<T>, arg));
                result = server.request_result(id);
                out << arg << ' ' << result << '\n';
            }
            break;
        case task::pow:
            out.open("pow.txt");
            for(int i = 0; i < number_works; i++){
                double arg = ((double)rand() / RAND_MAX) * 10;
                id = server.add_task(std::bind(fun_pow<T>, 2.0, arg));
                result = server.request_result(id);
                out << arg << ' ' << result << '\n';
            }
            break;
    }
}

template<typename T>
void tests(task t, std::string output){
    std::ifstream in(output);
    T arg, result;
    switch(t){
        case task::sin:
            while(! in.eof()) {
                in >> arg;
                in >> result;
                assert(fabs(result - sin(arg)) < 1);
            }
            std::cout << "Sin: " << "Test passed successfully!" << std::endl;
            break;
        case task::sqrt:
            while(! in.eof()) {
                in >> arg;
                in >> result;
                assert(fabs(result - sqrt(arg)) < 1);
            }
            std::cout << "Sqrt: " << "Test passed successfully!" << std::endl;
            break;
        case task::pow:
            while(! in.eof()) {
                in >> arg;
                in >> result;
                assert(fabs(result - pow(2.0, arg)) < 1);
            }
            std::cout << "Pow: " << "Test passed successfully!" << std::endl;
            break;
    }
}

int main()
{
    int number_works = 1000;
    task sin = task::sin;
    task sqrt = task::sqrt;
    task pow = task::pow;

    Server<double> server;

    std::jthread server_thread(&Server<double>::start, server);
    std::thread client_1(client<double>, std::ref(server), std::ref(sin), std::ref(number_works));
    std::thread client_2(client<double>, std::ref(server), std::ref(sqrt), std::ref(number_works));
    std::thread client_3(client<double>, std::ref(server), std::ref(pow), std::ref(number_works));

    client_1.join();
    client_2.join();
    client_3.join();
    server.stop(server_thread);
    server_thread.join();
    std::cout << "End\n";
    tests<double>(sin, "sin.txt");
    tests<double>(sqrt, "sqrt.txt");
    tests<double>(pow, "pow.txt");
}
