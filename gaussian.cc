#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdint>

#include "graphs.hpp"

union FloatInt {
    float f;
    uint32_t i;
};

namespace hack {
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
inline constexpr float fexp(float x) {
    return l2f(static_cast<unsigned long>(x * (1/epsilon() + 0x38aa22)) + f2l(1));
}

inline constexpr float flog(float x) {
    return static_cast<float>(epsilon() * 0.6931471805599453 * (f2l(x) - f2l(1) + 0x66774));
}
}


float refine_gaussian(float x, float initial_guess) {
    float x2 = x * x;
    float y = initial_guess;

    // h(y) = ln(y) + x²/2
    // h'(y) = 1/y
    // y_{n+1} = y_n - h(y_n)/h'(y_n)
    // = y_n - y_n(ln(y_n) + x²/2)
    // = y_n(1 - ln(y_n) - x²/2)

    // One iteration
    y = y * (1.0f - hack::flog(y) - 0.5f * x2);

    return y;
}

float refine_gaussian_v2(float x, float initial_guess) {
    // Work in log space to avoid overflow
    float x2 = x * x;
    float log_y = hack::flog(initial_guess);
    // log(y_new) = log(y) + log(1 - log(y) - (x*x)/2)
    float correction = 1.0f - log_y - 0.5f * x2;
    if (correction <= 0) return 0.0f; 
    return initial_guess * correction;
}


float fast_gaussian(float x) {
    union FloatInt u;

    uint32_t exp = (uint32_t)(-6051101.5f * x * x); // (1 << 23) * 0.5 * log2(e)
    u.i = (127 << 23) + exp;
    float y = u.f;

    return y;
}

float fast_gaussian_v2(float x) {
    return hack::fexp(-0.5f * x * x);
}


void test_gaussian() {
    int n_tests = 10;

    for (float x = 0.0f; x <= 3.0f; x+= 0.1f) {
        float true_value = std::exp(-x * x / 2.0f);
        float approx_value = fast_gaussian(x);
        float approx_value2 = fast_gaussian_v2(x);
        float error = true_value - approx_value;
        float error2 = true_value - approx_value2;

        std::cout << std::setprecision(15) 
            << "Input: " << x << "\n True: " << true_value 
            << "\n Fast: " << approx_value 
            << "\nFast2: " << approx_value2
            << "\n  Err: " << error 
            << "\n Err2: " << error2
            << "\n----" << std::endl;
    }
}


int main()
{


	size_t height = 160;
	size_t width = 160;

	double xmin = -3;
	double xmax = 3;
	double ymin = 0;
	double ymax = 1.0;
    
    const auto tg = [](auto x) { return std::exp(-x * x / 2.0f); };
    const auto fg = [](auto x) { return fast_gaussian(x); };
    const auto fg2 = [](auto x) { return fast_gaussian_v2(x); };
	std::function<float(float)> functions[] = {tg, fg, fg2};

	graphs::functions(height, width, xmin, xmax, ymin, ymax, 3, functions);

    test_gaussian();
	return 0;
}


