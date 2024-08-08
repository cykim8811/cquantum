
#include <stdio.h>

int main() {
    int a = 0;
    int b = 1;

    for (int i = 0; i < 10; i++) {
        int temp = a;
        a = b;
        b = temp + b;
    }

    printf("a: %d\n", a);
    printf("b: %d\n", b);

    return 0;
}

