#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def build_interpolated_lut(bits=4):
    n_entries = 1 << bits
    x_lut = np.linspace(0, 1, n_entries + 1)
    y_lut = np.sin(x_lut * np.pi / 2)
    table = y_lut[:-1]
    slopes = np.diff(y_lut)
    
    return table, slopes

def approx_sin_interpolated(x, table, slopes, bits=4):
    n_entries = 1 << bits
    scaled_x = x * n_entries
    indices = np.floor(scaled_x).astype(int)
    indices = np.clip(indices, 0, n_entries - 1)
    fractions = scaled_x - indices
    
    return table[indices] + fractions * slopes[indices]

bits = 4
table, slopes = build_interpolated_lut(bits)

x_test = np.linspace(0, 1, 10000)
y_true = np.sin(x_test * np.pi / 2)
y_approx_interp = approx_sin_interpolated(x_test, table, slopes, bits)

def approx_sin_basic(x, table, bits=4):
    n_entries = 1 << bits
    indices = (x * n_entries).astype(int)
    indices = np.clip(indices, 0, n_entries - 1)
    return table[indices]

y_approx_basic = approx_sin_basic(x_test, table, bits)

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_true, label='Exact $\sin(x \cdot \pi/2)$', color='black', lw=2)
plt.plot(x_test, y_approx_basic, label='4-bit Basic LUT (Staircase)', linestyle=':', color='blue')
plt.plot(x_test, y_approx_interp, label='4-bit Interpolated LUT', linestyle='--', color='red')
plt.title('Sine Approximation: Basic vs. Interpolated LUT (4-bit)')
plt.xlabel('Normalized Input $x \in [0, 1)$')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.savefig('trig_lut_comparison.png')

err_basic = np.abs(y_approx_basic - y_true) / y_true
err_interp = np.abs(y_approx_interp - y_true) / y_true

plt.figure(figsize=(10, 6))
plt.semilogy(x_test, err_basic, label='Basic LUT Error', color='blue', alpha=0.7)
plt.semilogy(x_test, err_interp, label='Interpolated LUT Error', color='red')
plt.title('Relative Error: Basic vs. Interpolated (Log Scale)')
plt.xlabel('Normalized Input $x$')
plt.ylabel('Absolute Error $|approx - true|$')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()
