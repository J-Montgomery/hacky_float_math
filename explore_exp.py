#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math


def gradient_descent(f, initial_x, step_size=0.01, epsilon=1e-6, max_iterations=1000):
    """
    Minimize error by adjusting x using gradient descent.

    Args:
        f: Function that returns an error value
        initial_x: Starting x value
        step_size: How far to move x in each iteration
        epsilon: Stop when error change is smaller than this
        max_iterations: Maximum number of iterations

    Returns:
        float: The x value that minimizes the error
    """
    x = initial_x

    for i in range(max_iterations):
        # Get current error
        error = f(x)

        # Estimate gradient using small perturbation
        h = 1e-1
        gradient = (f(x + h) - error) / h

        # Update x - move opposite to gradient to minimize error
        x_new = x - step_size * gradient

        # Calculate new error
        error_new = f(x_new)

        #print(f"Iteration {i}: x = {x:.6f}, error = {error:.6f}, update = {step_size * gradient}")

        # Check if we've improved enough
        if abs(error) < epsilon:
            break

        x = x_new

    return x

def incremental_search(f, initial_x, end, step_size=0.1, epsilon=0.01):
    x = initial_x
    best_x = x
    best_error = f(x)

    for i in np.arange(initial_x, end, step_size):
        x += step_size
        error = f(x)

        print(f"Iteration {i}: x = {x:.6f}, error = {error:.6f}")
        if error < best_error:
            best_x = x
            best_error = error

        if error < epsilon:
            print("Error below epsilon, stopping early")
            break

    return best_x

def exp_line(x, a, b):
    x = x.astype(np.float32)
    x = np.exp(a) + (np.exp(b) - np.exp(a)) / (b - a) * (x - a)
    return x

def broadcast(value, size):
    return value * np.ones(size)

def estimate_max_error(a, b):
    return (1/8)*((b - a)**2)*np.exp(a)

def estimate_max_error_location(a, b):
    return ((np.exp(a)*(1 + b - a))/np.exp(b)) + a + 1
    #return np.log((np.exp(b) - np.exp(a)) / (b - a))

def bump_curve(x, offset, amplitude, period):
    x = x.astype(np.float32)
    x -= offset
    x *= period
    return 4 * amplitude * np.mod(x, 1) * (1 - np.mod(x, 1))

def fast_exp(x):
    log2e = 12102203.0
    bias = 1065353216.0 # optimized for log2e = 12102204

    x = x.astype(np.float32) * 4
    y = (log2e*x+bias).astype(np.int32).view(np.float32)

    correction = bump_curve(x, offset = 3.444592, amplitude = 6.118131, period = 1.4427)

    # Correct for the supremum bias using our bump curve
    z = y - y * (correction/100)

    return np.sqrt(np.sqrt(z))

def explore_exp():
    start = 0.0
    end = 10.0
    x = np.linspace(start, end, 1000)

    true_curve = np.exp(x)
    fast_approx = fast_exp(x)
    relative_error = 100*(np.abs(fast_approx - true_curve) / true_curve)
    
    absolute_error = np.abs(fast_approx - true_curve)
    avg_relative_error = broadcast(np.mean(relative_error), relative_error.size)

    estimated_max_error = (1/8)*((end - start)**2)*np.exp(start)
    est_max_error = broadcast(estimate_max_error(start, end), relative_error.size)

    # bump_curve(x, offset = 3.444592, amplitude = 6.118131, period = 1.4427)
    bumps = bump_curve(x, offset = 3.444592, amplitude = 6.118131, period = 1.4427)
    

    print(f"estimated max error: {estimated_max_error}")
    print(f"estimated max error location: {estimate_max_error_location(start, end)}")
    print(f"Average error: {np.mean(relative_error)}")
    print(f"average absolute error: {np.mean(absolute_error)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(x, true_curve, 'b-', label = 'reference')
    ax1.plot(x, fast_approx, 'r--', label = 'approximation')
    ax1.set_xlabel('x')
    ax1.legend()
    ax1.grid(True)

    vals = np.convolve(np.abs(relative_error - bumps), np.ones(5)/5, mode='same')
    #ax2.plot(x, vals, 'r--', label = 'smoothed error')
    ax2.plot(x, relative_error, 'b-', label = '% relative error')
    #ax2.plot(x, absolute_error, 'g--', label = 'absolute error')
    ax2.plot(x, avg_relative_error, 'r--', label = 'average relative error')
    #ax2.plot(x, bumps, 'y--', label = 'bumps')
    #ax2.plot(x, relative_error - bumps, 'o--', label = 'bumps')
    ax2.legend()

    ax2.grid(True)

    # error_period = np.fft.fft2(relative_error)
    # print(f"error period: {np.max(error_period)}")
    plt.show()

def period_opt_func(value):
    start = 0.0
    end = 2.0
    x = np.linspace(start, end, 1000)
    true_curve = np.exp(x)
    fast_approx = fast_exp(x)
    relative_error = 100*(np.abs(fast_approx - true_curve) / true_curve)
    bump = bump_curve(x, offset = value, amplitude = 6.118131, period=1.4427)
    error = abs(relative_error - bump)
    error = np.convolve(error, np.ones(5)/5, mode='same')
    return np.mean(error)

def optimize_period():
    period = incremental_search(period_opt_func, 3.3, step_size = 1e-6, epsilon=0.001, end=3.7)
    print(f"found: {period}")

if __name__ == '__main__':
    explore_exp()
    #