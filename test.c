
#include <stdio.h>


int res = 0;

int main() {
    int a[5] = {1, 2, 3, 4, 5};
    int *b = &a;

    *b = 10;

    return 0;
}

