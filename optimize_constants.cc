#include <cmath>
#include <limits>
#include <vector>
#include <bit>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <cstdint>

#include <cassert>

float reference_impl(float x) {
    return expf(x);
}

float test_impl(float x, float scalar, float bias) {
    return std::bit_cast<float>((uint32_t)(std::fma(scalar, x , bias)));
}

float rel_error(float scalar, float bias, float x) {
    float true_value = reference_impl(x);
    float approx_value = test_impl(x, scalar,bias);
    auto rel_err = fabsf(true_value - approx_value) / true_value;
    return rel_err;
}

float average_rel_error(float scalar, float bias) {
    float total = 0.0f;
    float ctr = 0;
    for(float x = -88.0f; x <= 88.0f; x += 0.1f) {
        ctr += 1.0f;
        total += rel_error(scalar, bias, x);
    }
    return total / ctr;
}

// Grid search followed by local optimization
float optimize_magic(float initial_magic, float end) {
    // First do a grid search around the initial value
    float best_magic = initial_magic;
    const float scalar = 12102203.2f;
    float best_error = 1.0; // average_rel_error(scalar, initial_magic);

    std::size_t ctr = 0;
    std::size_t print_ctr = 1.00;
    for (float test_constant = initial_magic; test_constant <= end; test_constant = std::nextafterf(test_constant, INFINITY)) {

        float error = average_rel_error(scalar, test_constant);
        if (!std::isnan(error) && error < best_error) {
            printf("Grid search: magic=%f error=%e\n", test_constant, error);
            best_error = error;
            best_magic = test_constant;
        }
        ctr++;
        //std::cout << std::setprecision(30) << "testing magic: " << std::dec << test_constant << std::endl;
        if(ctr / print_ctr >= 10.0) {
            print_ctr = ctr;
            std::cout << std::setprecision(15) 
                      << "ctr: " << std::dec << ctr 
                      << " Best so far: " << best_magic 
                      << " error: " << best_error 
                      << " constant: " << test_constant << "\n";
        }
    }

    printf("Search result: magic=%f error=%e\n", best_magic, best_error);
    return best_magic;
}

int main() {
    // 12102203.161561485f -> optimal?
    // 1064872507.1541044f
    // float magic = 12102203.2f; // approx. (2^23) * log2(e)
    // float integer_1 = 0x3f800000; // 1.0f, converted to a float

    const float test1 = 1064872507.1541044f;
    const float test2 = 1064872507.0f;
    assert(test1 != test2);
    std::cout << std::setprecision(15) << "test1: " << test1 << " test2: " << test2 << std::endl;
    std::cout << std::setprecision(15) << "test1: " << std::nextafterf(test1, INFINITY) << " test2: " << std::nextafterf(test2, INFINITY) << std::endl;
    
    float initial_magic = 0x3f700000;
    float optimized_magic = optimize_magic(initial_magic, std::numeric_limits<float>::max());


    const float scalar = 12102203.2f;
    printf("Original magic: %f\n", initial_magic);
    printf("Optimized magic: %f\n", optimized_magic);
    printf("Original average error: %e\n", average_rel_error(scalar, initial_magic));
    printf("Optimized average error: %e\n", average_rel_error(scalar, optimized_magic));

    float test_points[] = {0.0f, 0.5f, 1.0f, 2.0f};
    for (float x : test_points) {
        float true_val = reference_impl(x);
        float approx_val = test_impl(x, scalar, optimized_magic);
        printf("x=%.1f: true=%f approx=%f error=%e\n", 
               x, true_val, approx_val, fabsf(true_val - approx_val));
    }

    return 0;
}