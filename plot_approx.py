#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math


def optimize_exp(x, bias):
    log2e = 12102204.0
    #bias = 1065353216.0 

    x = x.astype(np.float32)
    x = (log2e*x+bias).astype(np.int32)

    x = x.view(np.float32)
    return x

def exp_div(x):
    constant1 = 3248660424278399
    constant2 = 0x3fdf127e83d16f12

    x = x.astype(np.float64)
    hi = (constant1*x + constant2).astype(np.int64)
    lo = (constant2 - constant1*x).astype(np.int64)
    y = hi.view(np.float64) / lo.view(np.float64)
    return y.astype(np.float32)

## Refine the original fast_exp with 1 iteration of newton-raphson
def fast_exp_newton(x):
    log2e = 12102204.0
    bias = 1064809184.0 # optimized for log2e = 12102204
    #bias = 1064986464.0 # rel err bounded to 3%, log2e = 12102204

    x = x.astype(np.float32)
    y = (log2e*x+bias).astype(np.int32).view(np.float32)
    y = y * ( 2 - fast_log(y)/x)
    #y = y * ( 2 - fast_log(y)/x)
    #y = y * ( 2 - fast_log(y)/x)
    return y

    # float log(float x) {
    #     const float magic_scale = 8.26295831757307e-08f; // float epsilon * ln(2)
    #     const int32_t offset = std::bit_cast<int32_t>(1.0f); // 0x3f800000
    #     int32_t i = std::bit_cast<int32_t>(x);
    #     return magic_scale * (float)(i - offset);
    # }

def fast_exp_sqrt_range_reduction(x):
    log2e = 12102204.0
    bias = 1064809184.0 

    x = x.astype(np.float32)
    k = np.floor(x).astype(np.int32)
    y = x - k
    x = (log2e*2*y + bias).astype(np.int32)

    return np.sqrt(x.view(np.float32)) * np.exp(k)# Fantastic, but shrinks the range due to intermediate values


def fast_exp_sqrt(x):
    log2e = 24204416.0
    #bias = 1064809704.0 
    bias = 1064989414.0

    x = x.astype(np.float32)
    x1 = x
    x = (log2e*x + bias).astype(np.int32)

    y = np.sqrt(x.view(np.float32)) # Fantastic, but shrinks the range due to intermediate values
    #y = y * ( 2 - np.log(y)/x1)
    return y

def optimize_exp_mask(x, bias):
    x = x.astype(np.float32)
    x = x * 4
    y = fast_exp_sqrt(x)
    z = np.sqrt(np.sqrt(y)).view(np.int32)
    z = z - bias
    return z.view(np.float32)

def fast_exp_algebraic(x):
    x = x.astype(np.float32)
    x = x * 4
    y = fast_exp_sqrt(x)
    z = np.sqrt(np.sqrt(y)).view(np.int32)
    z = z - 4000
    return z.view(np.float32)

def fast_exp(x):
    log2e = 12102203.0 # original
    #log2e = 12102204.0 # optimized log2e
    bias = 1065353216.0 #original
    # bias = 1064631197.0 # ankerl w/ john correction
    #bias = 1064822049.0 # optimized 
    # bias = 1065323487.0
    #bias = 1064878369.0 # optimized for error centered around true value exp(x)
    
    #bias = 1064809184.0 # optimized for log2e = 12102204
    #bias = 1064986464.0 # rel err bounded to 3%, log2e = 12102204

    x = x.astype(np.float32)
    x = (log2e*x+bias).astype(np.int32)

    x = x.view(np.float32)
    return x

    # float log(float x) {
    #     const float magic_scale = 8.26295831757307e-08f; // float epsilon * ln(2)
    #     const int32_t offset = std::bit_cast<int32_t>(1.0f); // 0x3f800000
    #     int32_t i = std::bit_cast<int32_t>(x);
    #     return magic_scale * (float)(i - offset);
    # }

def anders_exp(x):
    INVLOG_2 = 1.442695040 # 1/ln(2)
    BIT_SHIFT = 8388608.0 # 2 ^ 23
    C1 = 121.2740838
    C2 = 27.7280233
    C3 = 4.84252568
    C4 = 1.49012907

    p = INVLOG_2 * x # x/ln(2)
    z = p - np.floor(p) # z = fractional part of p

    # Polynomial approximation of 2^x
    # (1/(C3-z))
    rcp = 1.0 / (C3 - z)
    rcp = rcp * C2 + (C1 + p)
    rcp = rcp - C4 * z

    result = (BIT_SHIFT * rcp).astype(np.int32).view(np.float32)

    return result
    
def fast_log(x):
    ln2epsilon = 8.26295831757307e-08
    bias = 0x3f800000 #127 * 2^23
    #ln2epsilon = 8.262958405176314e-8
    #bias = 1064866805
    x = x.astype(np.float32)
    x = x.view(np.int32)
    #x = x.view(np.int32)
    x = ln2epsilon * (x - bias).astype(np.float32)
    return x

def fast_exp_low(x):
    log2e = 12102203.0
    bias = 1064673119.0

    x = x.astype(np.float32)
    x = (log2e*x+bias).astype(np.int32)

    x = x.view(np.float32)
    return x

# def fast_exp(x):
#     log2e = 12102203.0
#     # bias = 1065353216.0
#     # bias = 1064631197.0
#     bias = 1064822049.0

#     x = x.astype(np.float32)
#     x = (log2e*x+bias).astype(np.int32)

#     x = x.view(np.float32)
#     return x

# def fast_exp(x):
#     x = x.astype(np.float32)
#     y = (6051102 * x + 1056478197).astype(np.int32)
#     z = (1056478197 - 6051102 * x).astype(np.int32)
#     return y.view(np.float32) / z.view(np.float32)

def compute_avg_rel_error(bias):
    x = np.linspace(1.0, 2.0, 10000)
    true_inverse = np.exp(x)
    fast_inverse = optimize_exp_mask(x, bias)
    relative_error = 100*(np.abs(fast_inverse - true_inverse) / true_inverse)
    return np.mean(relative_error)

def compute_min_rel_error(bias):
    x = np.linspace(1.0, 3.0, 1000)
    true_inverse = np.exp(x)
    fast_inverse = optimize_exp(x, bias)
    relative_error = 100*(np.abs(fast_inverse - true_inverse) / true_inverse)
    return np.max(relative_error)

def compare_exp():
    x = np.linspace(1.0, 2.0, 10000)

    true_inverse = np.exp(x)
    fast_inverse = fast_exp_algebraic(x)
    relative_error = 100*(np.abs(fast_inverse - true_inverse) / true_inverse)
    absolute_error = np.abs(fast_inverse - true_inverse)
    avg_relative_error = np.empty(relative_error.size)
    avg_relative_error.fill(np.mean(relative_error))
    print(f"Average error: {np.mean(relative_error)}")
    print(f"average absolute error: {np.mean(absolute_error)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(x, true_inverse, 'b-', label = 'reference')
    ax1.plot(x, fast_inverse, 'r--', label = 'approximation')
    ax1.set_xlabel('x')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, relative_error, 'b-', label = '% relative error')
    #ax2.plot(x, absolute_error, 'g--', label = 'absolute error')
    ax2.plot(x, avg_relative_error, 'r--', label = 'average relative error')
    ax2.legend()

    # minorLocator = MultipleLocator(0.25)
    # ax2.yaxis.set_minor_locator(minorLocator)
    ax2.grid(True)

    plt.show()

def compare_log():
    x = np.linspace(1.9999, 2.0, 10000)

    true_inverse = x
    fast_inverse = fast_exp(fast_log(x))
    relative_error = 100*(np.abs(fast_inverse - true_inverse)/true_inverse)
    absolute_error = np.abs(fast_inverse - true_inverse)
    avg_relative_error = np.empty(relative_error.size)
    avg_relative_error.fill(np.mean(relative_error))
    print(f"Average error: {np.mean(relative_error)}")
    print(f"average absolute error: {np.mean(absolute_error)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(x, true_inverse, 'b-', label = 'reference')
    ax1.plot(x, fast_inverse, 'r--', label = 'approximation')
    ax1.set_xlabel('x')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, relative_error, 'b-', label = '% relative error')
    #ax2.plot(x, absolute_error, 'g--', label = 'absolute error')
    ax2.plot(x, avg_relative_error, 'r--', label = 'average relative error')
    ax2.legend()
    ax2.grid(True)

    plt.show()

def optimize_exp_bias():
    bias = 0
    best_bias = bias
    best_error = compute_avg_rel_error(best_bias)
    ctr = 0
    while(bias <= (2**32)):
        ctr += 1
        rel_err = compute_avg_rel_error(bias)
        print(f"Testing {bias} {rel_err}")
        if(rel_err < best_error):
            print(f"New bias: {bias} err: {rel_err} prev: {best_error}")
            best_error = rel_err
            best_bias = bias
        if((ctr % 100000) == 0):
            print(f"testing {bias} {rel_err}")
        bias += 1 #int(100*math.log10(rel_err))+1
    print(f"Best bias {bias} {best_error}")

def optimize_exp_bias_negative():
    bias = 1064311000.0
    best_bias = bias
    best_error = compute_avg_rel_error(best_bias)
    ctr = 0
    while(bias <= 1067000000.0):
        ctr += 1
        max_err = compute_min_rel_error(bias)
        if(abs(max_err) < abs(best_error)):
            print(f"New bias: {bias} err: {max_err} prev: {best_error}")
            best_error = max_err
            best_bias = bias
        if((ctr % 1000000) == 0):
            print(f"testing {bias} {max_err}")
        bias += 10 # int(10*math.log10(rel_error))+1
    print(f"Best bias {bias} {best_error}")

if __name__ == '__main__':
    #compare_exp()
    compare_log()
    #optimize_exp_bias()
    #optimize_exp_bias_negative()