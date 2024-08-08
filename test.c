
#include <stdio.h>

char* res;


int main() {
    
    int a = 0;
    int b = 1;
    
    int temp = 0;
    for (int i = 0; i < 10; i++) {
        temp = a;
        a = b;
        b = temp + b;
    }

    printf("str %s str", "hi");

    return 0;
}

