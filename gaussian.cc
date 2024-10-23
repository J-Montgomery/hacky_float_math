#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <bit>

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
inline float fexp(float x) {
    uint32_t magic = 1/epsilon() + 0x38aa22;
    uint32_t magic_1 = f2l(1);
    std::cout << std::hex << "magic: " << magic << ", magic_1: " << magic_1 << std::endl;
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
    float magic = -6051101.5f; // (1 << 23) * 0.5 * log2(e) * -1
    float integer_1 = 0x3f800000; // 1.0f, converted to a float
    uint32_t i = (uint32_t)(magic * x * x + integer_1);

    return std::bit_cast<float>(i);
}

float fast_gaussian_v2(float x) {
    return hack::fexp(-0.5f * x * x);
}

float fast_gaussian_v3(float x) {
    // not correct
    // This computes exp(x^2), but we want exp(-x^2/2)
    const uint32_t sq_magic = 0xc085c000;
    const uint32_t exp_magic = 0xb8aa22;
    const uint32_t magic_1 = 0x3f800000; // std::bit_cast<uint32_t>(1.0f);
    // value is (sq_magic + 2*x) * exp_magic + magic_1
    const uint32_t magic_offset = sq_magic * exp_magic + magic_1;
    const uint32_t magic_factor = 2 * exp_magic;

    uint32_t i = std::bit_cast<uint32_t>(x);
    i = magic_factor * i + magic_offset;
    //i = (sq_magic + 2*i) * exp_magic + magic_1;
    float y = std::bit_cast<float>(i);
    return y;
}

float fast_gaussian_v4(float x) {
    // computes -x^2/2
    const uint32_t sq_magic = 0x4005c000;
    const uint32_t exp_magic = 0xb8aa22;
    const uint32_t magic_1 = 0x3f800000; // std::bit_cast<uint32_t>(1.0f);
    // value is (sq_magic + 2*x) * exp_magic + magic_1
    const uint32_t magic_offset = sq_magic * exp_magic + magic_1;
    const uint32_t magic_factor = 2 * exp_magic;

    uint32_t i = std::bit_cast<uint32_t>(x);
    i = (sq_magic + 2 * i);
    float y = std::bit_cast<float>(i);
    y = std::bit_cast<float>(std::bit_cast<uint32_t>(y * exp_magic) + magic_1);
    return y;
    //return hack::fexp(y);
}

float evil_square(float x) {

    // const uint32_t magic = -1u * (std::bit_cast<uint32_t>(1.0f) - 0x5c000);
    const uint32_t magic = 0xc085c000;
    std::cout << std::hex << magic << std::endl;

    uint32_t i = std::bit_cast<uint32_t>(x);
    // // Evil floating point bit hack
    // i += 0xFF800000;
    i  = magic + 2 * i;
    float y = std::bit_cast<float>(i);

    return 2 * x * std::sqrt(y) - y;
}

float evil_half_negative_square(float x) {
    // compute -x^2 / 2
    const uint32_t magic = -1u * (std::bit_cast<uint32_t>(-2.0f) - 0x5c000);
    //const uint32_t magic = 0xc085c000;
    std::cout << std::hex << magic << std::endl;

    uint32_t i = std::bit_cast<uint32_t>(x);
    // // Evil floating point bit hack
    // i += 0xFF800000;
    i  = magic + 2 * i;
    float y = std::bit_cast<float>(i);

    return y;
}

void test_square() {
 // calculation should be be of the form (magic + 2*x)
 // magic = (1-m) * (1 + bias) for x^m (m=2)
 // bias = -0x5c000
    for (float x = 0.0f; x <= 30.0f; x+= 3.14159f) {
        float true_square = (x * x) / -2.0f;
        float evil_sq = evil_half_negative_square(x);

        std::cout << std::setprecision(15)
            << "x: " << x << ", func: " << true_square
            << ", evil: " << evil_sq
            << ", error: " << true_square - evil_sq << std:: endl;
    }
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
	double ymin = -1.0;
	double ymax = 1.0;
    
    const auto tg = [](auto x) { return std::exp(-x * x / 2.0f); };
    const auto fg = [](auto x) { return fast_gaussian(x); };
    const auto fg2 = [](auto x) { return fast_gaussian_v2(x); };
	std::function<float(float)> functions[] = {tg, fg};

	graphs::functions(height, width, xmin, xmax, ymin, ymax, 2, functions);

    test_gaussian();

    //test_square();
	return 0;
}


