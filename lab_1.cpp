#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>

#ifdef TYPE
    typedef double type;
#else
    typedef float type;
#endif

using namespace std;

type arr[10000000];

int main(){
    type sum = 0;
    for(int i = 0; i < 10000000; i++){
        arr[i] = sin(i * M_PI * 2 / 10000000);
        sum += arr[i];
    }
    cout << sum << endl;
    return 0;
}