#include <cmath>
#include <limits>
#include <vector>
#include <bit>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <cstdint>

float true_gaussian(float x) {
    return expf(x);
}

float test_gaussian(float x, float magic) {
    float integer_1 = 0x3f800000;
    return std::bit_cast<float>((uint32_t)(std::fma(magic, x , integer_1)));
}

float calculate_error(float magic, float x_min = -4.0f, float x_max = 4.0f, int samples = 100) {
    float total_error = 0.0f;

    float step = (x_max - x_min) / samples;
    for (float x = x_min; x <= x_max; x += step) {
        float true_value = true_gaussian(x);
        float approx_value = test_gaussian(x, magic);
        total_error += fabsf(true_value - approx_value);
    }

    return total_error / samples;
}

// Grid search followed by local optimization
float optimize_magic(float initial_magic, float start, float end, float step) {
    // First do a grid search around the initial value
    float best_magic = initial_magic;
    float best_error = calculate_error(initial_magic);

    // Search in increasingly fine grids
    float search_ranges[] = {100000.0f, 10000.0f, 1000.0f, 100.0f, 10.0f, 1.0f};
    int points_per_range = 10;

    std::size_t ctr = 0;
    std::size_t print_ctr = 1.00;
    for (float test_magic = start; test_magic <= end; test_magic += step) {
        if(test_magic + step == test_magic) {
            test_magic = std::nextafterf(test_magic, INFINITY);
        }

        float error = calculate_error(test_magic);
        if (error < best_error) {
            printf("Grid search: magic=%f error=%e\n", test_magic, error);
            best_error = error;
            best_magic = test_magic;
        }
        ctr++;
        if(ctr / print_ctr >= 10.0) {
            print_ctr = ctr;
            std::cout << std::setprecision(15) << "ctr: " << ctr << " Best so far: " << best_magic << " error: " << best_error << "\n";
        }
    }

    printf("Grid search result: magic=%f error=%e\n", best_magic, best_error);
    return best_magic;
}

int main() {
    float initial_magic = 12100000.0f;
    float optimized_magic = optimize_magic(initial_magic, 0.0, std::numeric_limits<float>::max(), 1.0);

    printf("Original magic: %f\n", initial_magic);
    printf("Optimized magic: %f\n", optimized_magic);
    printf("Original average error: %e\n", calculate_error(initial_magic));
    printf("Optimized average error: %e\n", calculate_error(optimized_magic));

    float test_points[] = {0.0f, 0.5f, 1.0f, 2.0f};
    for (float x : test_points) {
        float true_val = true_gaussian(x);
        float approx_val = test_gaussian(x, optimized_magic);
        printf("x=%.1f: true=%f approx=%f error=%e\n", 
               x, true_val, approx_val, fabsf(true_val - approx_val));
    }

    return 0;
}