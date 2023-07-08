#include <iostream>
#include <string>

using namespace std;

void switch_int(int *test, int &a, int &b)
{
    *test = b;
}

int add_10(int *test)
{
    (*test) += 10;
    return *test;
}

// int add_10(int &test) {
//     test += 10;
//     return test;
// }

int main()
{
    int num = 1, num1 = 2;
    int *p = &num;
    cout << *p << endl;
    switch_int(p, num, num1);
    // after the swich_int function, the value of num is changed to 2.
    // but the address of p is not changed.
    cout << *p << endl;
    add_10(p);
    cout << p << (&num) << (&num1) << endl;
    cout << num << " " << num1 << endl;
    // why num1 is still 2, not 12?
    // the reason is that the function add_10 is a function of value, not a function of reference.
    // so the correct code is:
    // int add_10(int &test) {
    //     test += 10;
    //     return test;
    // }
    // and the result is:
    // 1
    // 2
    // 12
    // 12
    // so the function of value is not a good way to change the value of a variable.
    return 0;
}
