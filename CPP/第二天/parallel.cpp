#include <iostream>
#include <omp.h>   // NEW ADD

using namespace std;

int main()
{

        #pragma omp parallel for num_threads(4) // NEW ADD
        for(int i=0; i<10; i++)
        {
        cout << i << endl;
        }
        return 0;
}