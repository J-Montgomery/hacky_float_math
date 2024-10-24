#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <vector>
#include <algorithm>
#include <random>

#include <benchmark/benchmark.h>

#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("avx,avx2,fma"))), apply_to=function)
#endif
namespace fast {

    float exp(float x) {
        //float magic = 214760456192.0f; // more accurate for large values, loses significant precision for small values
        float magic = 12102203.2f; // approx. (2^23) * log2(e)
        float integer_1 = 0x3f800000; // 1.0f, converted to a float
        return std::bit_cast<float>((int32_t)(std::fma(magic, x, integer_1)));
    }

    __m256 avx2_exp_f32(__m256 x) {
        const __m256 magic = _mm256_set1_ps(12102203.2f);
        const __m256 offset = _mm256_set1_ps(0x3f800000);

        return _mm256_castsi256_ps(_mm256_cvttps_epi32(_mm256_fmadd_ps(magic, x, offset)));
        return x;
    }


    float avx2_softmax_f32(const std::size_t n, float *y, float *x, float max) {
        std::size_t i = 0;
        float sum = 0;
        // We have a full 256 bit lane available
        for(; i + 7 < n; i += 8) {
            __m256 val = avx2_exp_f32(_mm256_sub_ps(_mm256_loadu_ps(x+i), _mm256_set1_ps(max)));
            _mm256_storeu_ps(y + i, val);

            // _mm256_reduce_add_ps() doesn't exist, so instead extract the results
            // 128 bits at a time
            __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                    _mm256_castps256_ps128(val));
            val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
            val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
            sum += (float)_mm_cvtss_f32(val2);
        }

        for (; i < n; ++i) {
            float val = fast::exp(x[i] - max);
            sum += val;
            y[i] = val;
        }

        return sum;
    }

    float vec_softmax(std::vector<float> input) {
        auto max_val = *std::max_element(input.begin(), input.end());
        decltype(input) output(input.size());
        return fast::avx2_softmax_f32(input.size(), output.data(), input.data(), max_val);
    }
}

float softmax(std::vector<float>& input) {
    auto max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    std::transform(input.begin(), input.end(), input.begin(),
    [max_val, &sum](float val) {
        float exp_val = std::exp(val - max_val);
        sum += exp_val;
        return exp_val;
    });

    float inv_sum = 1.0 / sum;
    std::transform(input.begin(), input.end(), input.begin(),
        [inv_sum](float val) { return val * inv_sum; });

    return sum;
}
#if defined(__clang__)
#pragma clang attribute pop
#endif

std::vector<float> generate_random_data(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0, 10.0);

    std::vector<float> data(size);
    for (auto& val : data) {
        val = dis(gen);
    }
    return data;
}

static void BM_SoftmaxBasic(benchmark::State& state) {
    const size_t size = state.range(0);
    auto input = generate_random_data(size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(softmax(input));
    }
    state.SetItemsProcessed(state.iterations() * size);
}

static void BM_SoftmaxOptimized(benchmark::State& state) {
    const size_t size = state.range(0);
    auto input = generate_random_data(size);

    for (auto _ : state) {
        auto input_copy = input;
        benchmark::DoNotOptimize(fast::vec_softmax(input_copy));
    }
    state.SetItemsProcessed(state.iterations() * size);
}

BENCHMARK(BM_SoftmaxBasic)
    ->RangeMultiplier(2)
    ->Range(8, 8<<10)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SoftmaxOptimized)
    ->RangeMultiplier(2)
    ->Range(8, 8<<10)
    ->Unit(benchmark::kMicrosecond);