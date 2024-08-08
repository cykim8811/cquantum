
#include <stdio.h>

// char w;
int s = 10;
int k[3] = {1, 2, 3};
// float j = 11.2;
int *asd[3] = {&s, &s, &s};
int *q = &s;


int test(int* a, char b) {
    float e[3] = {1.1, 2.2, 3.3};
    e[1] = 2.3;
    return (int)e[2];
}

int main() {
    int a = 0;
    int b = 1;

    for (int i = 0; i < 2; i++) {
        a += b;
        int tmp = a;
        a = b;
        b = tmp;
    }

    test(&a, 'a');

    printf("res = %d\n", a);

    printf("Hello World!\n");
    char* c = "hi";

    return 0;
}

