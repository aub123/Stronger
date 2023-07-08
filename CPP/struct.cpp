#include <iostream>
#include <string>

using namespace std;
int main() {
    typedef struct list{
        int data;
        list * next;
    } *listptr;
    listptr head, p, q;
    head = new list;
    head->next = 0;
    p = head;
    for (int i = 1; i <= 10; i++) {
        q = new list;
        q->data = i;
        q->next = nullptr;
        p->next = q;
        p = q;
    }
    // print the list
    p = head;
    while (p) {
        cout << p->data << " ";
        p = p->next;
    }
    return 0;
}