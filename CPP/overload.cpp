#include <iostream>
#include <string>
using namespace std;

// int add(int a, int b) {
//     return a + b;
// }

// float add(float a, float b) {
//     return a + b;
// }

// int add(int a, float b) {
//     return a + b;
// }

// the above functions are examples of overload.
// but there is an another better way to do this:
// use the template function.
template <typename T>
T add(T a, T b) {
    return a + b;
}
// in this function , T is a type, it can be int, float, double, char, etc.
// and the types of a or b must be the same.
// you can use the function to add 1 and 2, like this:
// add<int>(1, 2);
// if they are not same, you can use the function like this:
// add<int, float>(1, 2.0);
// the result is 3.0
template <typename T, typename U>
T add(T a, U b) {
    return a + b;
}

// if you have a struct , how to use the template function:
struct person {
    string name;
    int age;
};
template <>
person add<person>(person a, person b) {
    person c;
    c.name = a.name + b.name;
    c.age = a.age + b.age;
    return c;
}


int main() {
    // a example of overload
    // the function of overload is that the function name is the same, but the parameters are different.
    // auto num = 1.8;
    // auto num1 = 2;
    // auto res =  add(num, num1);
    person zhang {"zhang", 18};
    person wang {"wang", 20};
    auto li = add(zhang, wang);
    // the error is that the type of num is double, the type of num1 is int.
    // you should use the same type.But if you want to use the different types:
    // cout <<res<< endl;
    cout << li.name << " " << li.age << endl;
}