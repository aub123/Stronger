#include <iostream>
using namespace std;

int main()
{

// test if openmp could run

#if _OPENMP
        cout << " support openmp " << endl;
#else
        cout << " not support openmp" << endl;
#endif
        return 0;
}