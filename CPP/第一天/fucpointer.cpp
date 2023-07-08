#include <iostream>
#include <string>
using namespace std;

//function pointer is used to point to a function.
//the requirements of the function pointer:
//1. the function pointer must point to a function.
//2. the function pointer must have the same return type as the function.
//3. the function pointer must have the same parameters as the function.
//4. the function pointer must have the same name as the function.
//5. the function pointer must have the same type as the function.
//6. the function pointer must have the same const as the function.


//the syntax is:
//return_type (*pointer_name)(parameter_list);
//for example:
int add(const int a,const int b) {
    return a + b;
}

int sub(int a, int b) {
    return a - b;
}



int main() {
    //create a function pointer
    int (*p)(int, int);
    // firstly point to add
    p = add;
    cout << p(1, 2) << endl;
    //secondly point to sub
    p = sub;
    cout << p(1, 2) << endl;
    //so the function pointer can be used to point to different functions.

    //also there is function reference:
    int (&q)(int, int) = add;
    cout << q(1, 2) << endl;
    //the function reference is the same as the function pointer.
    //but the function reference can not be changed.
    //so the function reference is more safe than the function pointer.
    return 0;
}