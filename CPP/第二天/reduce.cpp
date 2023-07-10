#include <iostream>
#include <omp.h> // NEW ADD

using namespace std;

int main()
{

    int sum = 0;
#pragma omp parallel for num_threads(32) reduction(+ : sum)
    /* 
    if "reduction(+ : sum)" is not be added,
    the result will be uncertained, 
    'cause the sum is shared by all threads, 
    when one thread is adding the sum,
    other threads may also add the sum,
    so the result will be different every time.
    The "reduction(+ : sum)" will make the sum private for each thread,
    and add them together after the loop.
    Alike, we can use "reduction(* : sum)" to get the product of all threads.
    */
    for (int i = 0; i < 100; i++)
    {
        sum += i;
        // print i and thread id
        cout << i << " " << omp_get_thread_num() << endl;
        // we can see that the thread id is not in order,
        // 'cause the threads are created in random order.
    }

    cout << sum << endl;
}