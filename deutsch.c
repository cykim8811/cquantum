
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "quantum.h"


rev void oracle(int8_t *output, int8_t *input) {
    *output ^= *input ^ 1;
}

int main() {
    int8_t* input  = malloc(1);
    int8_t* output = malloc(1);

    *input = 0;
    *output = 1;

    h(input, 1);
    h(output, 1);

    oracle(output, input);

    h(input, 1);

    if (*input) {
        printf("Constant function\n");
    } else {
        printf("Balanced function\n");
    }

    free(input);
    free(output);

    return 0;
}
