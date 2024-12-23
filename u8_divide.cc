#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <bit>

#include "graphs.hpp"


uint8_t correct_int_divide(float a, float b) {
    a = static_cast<uint8_t>(a);
    b = static_cast<uint8_t>(b);
    const uint8_t result = a / b;
    return static_cast<float>(result);
}

uint8_t fast_int_divide(float a, float b) {
    float inverse_b = exp2f(-0.99999*log2f(b));
    return a * inverse_b;
}

static uint32_t magic_constant = 0x7f000000;
//static float magic_scale = 1.10762596130371;
static float magic_scale2 = 1.00100016593933;
static float integer_1 = 1064782016; // 1.0f, converted to a float
static uint32_t magic_mask = 0x7eb504f3;
static float magic_number = 1.38535642623901;
static float approx_magic_number = 1.48927379686;

// float evil_inverse(float b) {
//     // We want exp2(-log2(b))
//     // This is equivalent to: (1/b)
//     // In IEEE 754, this can be done by negating the exponent
//     // Which is approximately: 0x7f000000 - bits(b)
//     b *= magic_scale;
//     float result = std::bit_cast<float>(magic_constant - (std::bit_cast<uint32_t>(b)));

//     return result;
// }

// uint8_t evil_divide(uint8_t a, uint8_t b) {
//     return a * evil_inverse(static_cast<float>(b));
// }

float approx_log(float x) {
    const float magic_scale = 1.192092896E-7; // float epsilon
    const int32_t offset = std::bit_cast<int32_t>(1.0f); // 0x3f800000
    int32_t i = std::bit_cast<int32_t>(x);
    return magic_scale * (float)(i - offset);
}

float approx_exp(float x) {

    float magic = 8388608;
    //float integer_1 = 0x3F7A8480; // 1.0f, converted to a float
    return std::bit_cast<float>((int32_t)(std::fma(magic, x, integer_1)));
}

float proper_divide2(uint8_t a, uint8_t b) {
    // Constants from the data section
    const float magic1 = std::bit_cast<float>(0xb4000000); // negative power of 2
    const float magic2 = std::bit_cast<float>(0x4b000000); // large power of 2
    const float magic3 = std::bit_cast<float>(0x4e7ddd23);

    // Convert inputs to float
    float fa = static_cast<float>(a);
    float fb = static_cast<float>(b);

    // movd + add operation: subtract 0x3f800000 (float 1.0) from bits of fb
    uint32_t temp = std::bit_cast<uint32_t>(fb) - 0x3f800000;

    // Convert back to float
    float ftemp = static_cast<float>(static_cast<int32_t>(temp));

    // Multiply by magic constants
    ftemp = ftemp * magic1 * magic2;

    // Add magic3
    ftemp = ftemp + magic3;

    // Convert to int and multiply by original a
    float result = static_cast<float>(static_cast<int32_t>(ftemp)) * fa;

    // Final conversion to integer
    return result;
}
// float reciprocal_1_f (float x){
//     int i = *(int*)&x;
//     i = 0x7eb504f3 - i;
//     float y = *(float*)&i;
//     y = 1.94285123*y*fmaf(-x, y, 1.43566f);
//     return y;
// }

float reciprocal_1_f (float x){
    int i = *(int*)&x;
    i = 0x7eb504f3 - i;
    float y = *(float*)&i;
    y = 1.94285123*y*fmaf(-x, y, 1.43566f);
    return y;
}

uint8_t magic_divide(uint8_t a, uint8_t b) {
    float x = static_cast<float>(b);
    int i = 0x7eb504f3 - *reinterpret_cast<int*>(&x);
    float reciprocal = std::bit_cast<float>(i);
    return a*1.94285123f*reciprocal*(-x*reciprocal+1.43566f);
}


uint8_t approx_magic_divide(uint8_t a, uint8_t b) {
    float x = static_cast<float>(b);
    int i = 0x7eb504f3 - *reinterpret_cast<int*>(&x);
    float reciprocal = std::bit_cast<float>(i);
    reciprocal = 1.385356f*reciprocal;
    return a*reciprocal;
}

size_t test_u8_divide() {
    size_t num_failed = 0; 
    for(int i = 0; i < (256*256); i++) {
        const float b = static_cast<float>((i >> 8) + 1);
        const float a = static_cast<float>(i & 0xFF);
        const float result = correct_int_divide(a, b);
        const float fast_result = magic_divide(a, b);
        if(result != fast_result) {
            std::cout << std::setfill('0') << std::setw(2) << static_cast<int>(a) << " / " << std::setw(2) << static_cast<int>(b) << " != " << std::setw(2) << static_cast<int>(result) << " " << std::setw(2) << static_cast<int>(fast_result) << "\n";
            num_failed++;
        }
    }

    return num_failed;
}

int main()
{
	size_t height = 160;
	size_t width = 160;

	double xmin = 0;
	double xmax = 255;
	double ymin = 0;
	double ymax = 10;
    
    const auto tg = [](auto x) { return correct_int_divide(255.0, x); };
    const auto fg = [](auto x) { return magic_divide(255, x); };
	std::function<float(float)> functions[] = {tg, fg};

	graphs::functions(height, width, xmin, xmax, ymin, ymax, 2, functions);
    size_t iterations = 0;
    size_t failures = 0;
    size_t min = 256*256;
    // while((failures = test_u8_divide()) > 0) {
    //     if(!(iterations % 1000)) {
    //         std::cout << "Trying " << std::setprecision(15) << std::hex <<approx_magic_number << std::dec << "\n";
    //         std::cout << "Failures: " << failures << "\n";
    //         std::cout << "Min: " << min << "\n";
    //     }

    //     // if(failures < min) {
    //     //     std::cout << "New min: " << std::setprecision(15) << magic_mask << " Failures: " << failures << std::dec << "\n";
    //     //     min = failures;
    //     // }
    //     iterations++;
    //     //magic_mask -= 1;
    //     approx_magic_number = std::nextafter(approx_magic_number, -std::numeric_limits<float>::infinity());
    // }
    std::cout << "Failures: " << test_u8_divide() << " with magic_scale2: " << std::setprecision(15) << magic_scale2 << "\n";
	return 0;
}
