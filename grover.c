
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "quantum.h"

rev void oracle(int8_t *output, int8_t *input) {
    // In the rev function, the condition must be satisfied even when exiting the if statement.
    if (*input == 15) {
        *output ^= 1;
    }
    // In the rev function, when freeing, it must be guaranteed that the memory pointed to by the pointer is 0
}

rev void diffuser(void* input, size_t n) {
    uint8_t* x = (uint8_t*) input;
    h(x, n);
    for (size_t i = 0; i < n; i++) {
        x[i] = ~x[i];
    }
    mcpf(x, n);
    for (size_t i = 0; i < n; i++) {
        x[i] = ~x[i];
    }
    h(x, n);
}

int main() {
    int8_t* input  = malloc(1);
    int8_t* output = malloc(1);

    *input = 0;
    *output = 1;

    h(output, 1);
    h(input, 1);
    
    int iter_count = (int) round(3.141592 / 4 * sqrt(1 << 8));
    for (size_t i = 0; i < iter_count; i++) {
        oracle(output, input);
        diffuser(input, 1);
    }

    printf("%d\n", *input);

    free(input);
    free(output);

    return 0;
}
