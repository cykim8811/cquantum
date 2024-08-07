
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "quantum.h"

rev void oracle(uint8_t *output, uint8_t *input, uint16_t target) {
    uint16_t res = (uint16_t)input[0] * input[1];
    if (res == target) {
        *output ^= 1;
    }
    res /= input[1];
    res -= input[0];
    // In the rev function, all local variables must be 0 when exiting the function
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

rev void grover(void (*oracle)(void*, void*, uint16_t), void* input, void* output, size_t n, uint16_t target) {
    int iter_count = (int) round(3.141592 / 4 * sqrt(1 << (n * 8)));
    for (size_t i = 0; i < iter_count; i++) {
        oracle(output, input, target);
        diffuser(input, n);
    }
}

uint8_t* guess(uint8_t target) {
    uint8_t* input = malloc(2);
    uint8_t* output = malloc(1);

    *input = 0;
    *output = 1;

    h(output, 1);
    h(input, 2);

    grover(oracle, input, output, 2, target);

    free(output);
    
    return input;
}

typedef struct {
    uint8_t a;
    uint8_t b;
    uint8_t count;
} factor;

int compare(const void* a, const void* b) {
    return ((factor*)a)->count - ((factor*)b)->count;
}

int main() {
    uint8_t target = 35;
    int shots = 1000;

    factor* guesses = malloc(shots * sizeof(factor));

    for (int i = 0; i < shots; i++) {
        uint8_t* res = guess(target);
        guesses[i].a = (res[0] > res[1]) ? res[1] : res[0];
        guesses[i].b = (res[0] > res[1]) ? res[0] : res[1];
        guesses[i].count = 1;
        free(res);
    }

    for (int i = 0; i < shots; i++) {
        for (int j = i + 1; j < shots; j++) {
            if (guesses[i].a == guesses[j].a && guesses[i].b == guesses[j].b) {
                guesses[i].count++;
                guesses[j].count = 0;
            }
        }
    }

    int available_guesses = 0;
    for (int i = 0; i < shots; i++) {
        if (guesses[i].a * guesses[i].b != target || (guesses[i].a == 1 || guesses[i].b == 1)) {
            guesses[i].count = 0;
        }
        if (guesses[i].count > 0) {
            available_guesses++;
        }
    }

    factor* factors = malloc(available_guesses * sizeof(factor));

    int j = 0;
    for (int i = 0; i < shots; i++) {
        if (guesses[i].count > 0) {
            factors[j].a = guesses[i].a;
            factors[j].b = guesses[i].b;
            factors[j].count = guesses[i].count;
            j++;
        }
    }

    free(guesses);

    qsort(factors, available_guesses, sizeof(factor), compare);

    for (int i = 0; i < available_guesses; i++) {
        printf("%d * %d = %d\n", factors[i].a, factors[i].b, target);
    }

    free(factors);

    if (available_guesses == 0) {
        printf("prime\n");
    }

    return 0;
}
