#include <iostream>
#include <string>
using namespace std;

int main() {
    char rabbit[6] = {'r', 'a', 'b', 'b', 'i', 't'};
    // a bad way to create a string, the end of the string is '\0'
    // the result is rabbitpig, because the end of the string is not '\0'
    char pig[6] = {'p', 'i', 'g'};
    // print the string, the result is pig, because the end of the string is '\0' ,the other characters are ignored

    char anotherpig[6] = {'p', 'i', 'g', '\0'};
    // a good way to create a string
    cout << rabbit <<" " << pig << endl;
    return 0;
}