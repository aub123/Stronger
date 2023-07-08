#include <iostream>
#include <string>

using namespace std;

void switch_str(char * test, char *a, char *b)
{
    test = b;
}
int main()
{
    char *rabbit = "rabbit";
    char *pig = "pig";
    cout << rabbit << endl;
    char *test = rabbit;

    switch_str(test, rabbit, pig);
    cout << test << endl;
    return 0;
}

// the above code is wrong, the data of test is not changed, it is still rabbit.
// because the pointer test is a local variable, 
// the data of test is a copy of rabbit, the data of rabbit is not changed.
// if you want to change the data of test , you should use the pointer of pointer, like this:
// void switch_str(char ** test, char *a, char *b)
// {
//     *test = b;
// }
// int main()
// {
//     char *rabbit = "rabbit";
//     char *pig = "pig";
//     cout << rabbit << endl;
//     char *test = rabbit;

//     switch_str(&test, rabbit, pig);
//     cout << test << endl;
//     return 0;
// }
// in this way , the data of test is changed,
// because the data of test is the address of rabbit, the data of rabbit is changed.

//constant pointer and pointer to constant

// int * const p2 = &num;
// *p2 = 10; // ok
// p2 = &num2; // error, p2 is a const pointer, the address of p2 is not changed.
// this is a example of constant pointer.
// const 的是 p2, 也就是指针（地址）

// int num = 1;
// int num2 = 2;
// const int * p1 = &num;
// const 的是 int，也就是不能修改int值
// *p1 = 10; // error, p1 is a pointer to const, the data of p1 is not changed.
// p1 = &num2; // ok, the address of p1 is changed.
// this is a example of pointer to constant.

// const int * const p3 = &num;
// *p3 = 10; // error, p3 is a pointer to const, the data of p3 is not changed.
// p3 = &num2; // error, p3 is a const pointer, the address of p3 is not changed.
// this is a example of constant pointer to constant.