#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <cmath>

union lens { float float_view; unsigned long int_view; };
union blens { unsigned long int_view; float float_view; };


// https://github.com/leegao/float-hacks/
inline constexpr unsigned long f2l(float f) {
    return (lens { f }).int_view;
}

inline constexpr float l2f(unsigned long x) {
    return (blens { x }).float_view;
}

inline constexpr float epsilon() {
    return l2f(f2l(1) + 1) - 1;
}

float flog(float x) {
    return static_cast<float>(epsilon() * 0.6931471805599453 * (f2l(x) - f2l(1) + 0x66774));
}
float func(float x, float curvature = 50000.0f) {
    return std::bit_cast<float>((uint32_t)(-0x3f800000 - curvature*x)) + 1.5*2.718281828f;
}

struct FuncWrapper {
    float curvature;
    static float func_with_curvature(float x, float c) {
        return std::bit_cast<float>((uint32_t)(-0x3f800000 - c*x)) + 1.5*2.718281828f;;
    }

    float operator()(float x) const {
        return func_with_curvature(x, curvature);
    }
};

/* -------------------------- */

template<typename F>
float second_derivative(F f, float x, float h = 0.0001f) {
    float fx_plus_h = f(x + h);
    float fx = f(x);
    float fx_minus_h = f(x - h);
    return (fx_plus_h - 2*fx + fx_minus_h) / (h*h);
}

// Helper function to calculate approximate first derivative
template<typename F>
float first_derivative(F f, float x, float h = 0.0001f) {
    return (f(x + h) - f(x - h)) / (2*h);
}

// Calculate approximate curvature using κ = |f''| / (1 + f'^2)^(3/2)
template<typename F>
float calculate_curvature(F f, float x) {
    float first_deriv = first_derivative(f, x);
    float second_deriv = second_derivative(f, x);

    return std::abs(second_deriv) / std::pow(1 + first_deriv*first_deriv, 1.5f);
}

// Function to calculate average curvature difference over the range
float curvature_difference(float curvature_param, float x_min, float x_max, int num_points) {
    FuncWrapper func_wrap{curvature_param};
    auto f = [func_wrap](auto x){func_wrap(x);};

    float total_diff = 0.0f;
    float step = (x_max - x_min) / (num_points - 1);

    for (int i = 0; i < num_points; ++i) {
        float x = x_min + i * step;
        float fun_curvature = calculate_curvature(func_wrap, x);
        float flog_curvature = calculate_curvature(flog, x);
        total_diff += std::abs(fun_curvature - flog_curvature);
    }

    return total_diff / num_points;
}

float optimize_curvature(float min_curvature, float max_curvature,
                        float tolerance = 1e-5f, int max_iterations = 100) {
    const float golden_ratio = (1.0f + std::sqrt(5.0f)) / 2.0f;
    const float x_min = 1.0f;
    const float x_max = 1000.0f;
    const int num_points = 10;  // Number of points to sample for curvature comparison

    float a = min_curvature;
    float b = max_curvature;
    float c = b - (b - a) / golden_ratio;
    float d = a + (b - a) / golden_ratio;

    for (int i = 0; i < max_iterations; ++i) {
        float fc = curvature_difference(c, x_min, x_max, num_points);
        float fd = curvature_difference(d, x_min, x_max, num_points);

        if (fc < fd) {
            b = d;
            d = c;
            c = b - (b - a) / golden_ratio;
        } else {
            a = c;
            c = d;
            d = a + (b - a) / golden_ratio;
        }

        if (std::abs(b - a) < tolerance) {
            return (a + b) / 2.0f;
        }
    }

    return (a + b) / 2.0f;
}

int main() {
    // Example usage
    float optimal_curvature = optimize_curvature(1000.0f, 100000.0f);
    printf("Optimal curvature parameter: %f\n", optimal_curvature);
    return 0;
}