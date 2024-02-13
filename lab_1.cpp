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
    for(int i = 1; i < 10000000; i++){
        arr[i - 1] = sin(i/(M_PI * 2));
        sum += arr[i - 1];
    }
    cout << sum << endl;
    return 0;
}