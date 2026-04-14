#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def build_pow_lut(bits_x=5, bits_y=5, x_range=(0.0, 1.0), y_range=(0.5, 4.0)):
    n_x = 1 << bits_x
    n_y = 1 << bits_y
    x_vals = np.linspace(x_range[0], x_range[1], n_x + 1)
    y_vals = np.linspace(y_range[0], y_range[1], n_y + 1)
    
    xv, yv = np.meshgrid(x_vals, y_vals, indexing='ij')
    table = np.power(xv, yv)
    return table, x_range, y_range

def approx_pow_2d(x_arr, y_arr, table, x_range, y_range, bits_x, bits_y):
    n_x, n_y = 1 << bits_x, 1 << bits_y
    
    sx = (x_arr - x_range[0]) / (x_range[1] - x_range[0]) * n_x
    sy = (y_arr - y_range[0]) / (y_range[1] - y_range[0]) * n_y
    
    ix = np.clip(np.floor(sx).astype(int), 0, n_x - 1)
    iy = np.clip(np.floor(sy).astype(int), 0, n_y - 1)
    
    fx, fy = sx - ix, sy - iy
    
    c00 = table[ix, iy]
    c10 = table[ix + 1, iy]
    c01 = table[ix, iy + 1]
    c11 = table[ix + 1, iy + 1]
    
    return (1 - fx) * (1 - fy) * c00 + fx * (1 - fy) * c10 + \
           (1 - fx) * fy * c01 + fx * fy * c11

def approx_log2_interpolated_2bit(x):
    mantissa, exponent = np.frexp(x)
    m = mantissa * 2.0
    e = (exponent - 1).astype(np.float32)
    
    table = np.array([
        0.00000000, # log2(1.00)
        0.32192809, # log2(1.25)
        0.58496250, # log2(1.50)
        0.80735492, # log2(1.75)
        1.00000000  # log2(2.00)
    ], dtype=np.float32)

    f = m - 1.0
    scaled_f = f * 4.0
    
    idx = np.floor(scaled_f).astype(int)
    idx = np.clip(idx, 0, 3) 
    
    fraction = scaled_f - idx

    y0 = table[idx]
    y1 = table[idx + 1]
    
    log2_m = y0 + fraction * (y1 - y0)
    
    return e + log2_m

def approx_exp2_interpolated_2bit(x):    
    exponent = np.floor(x).astype(int)
    fractional_part = x - exponent
    table = np.array([1.0, 1.1892, 1.4142, 1.6818, 2.0], dtype=np.float32)

    scaled_f = fractional_part * 4.0
    idx = np.clip(np.floor(scaled_f).astype(int), 0, 3)
    frac = scaled_f - idx
    
    y0 = table[idx]
    y1 = table[idx + 1]
    
    exp2_f = y0 + frac * (y1 - y0)

    return np.ldexp(exp2_f, exponent)

def approx_pow_via_log_exp(x, y):
    log2_x = approx_log2_interpolated_2bit(x)
    product = y * log2_x
    return approx_exp2_interpolated_2bit(product)

bits = 2
x_r, y_r = (0.01, 6.0), (0.5, 4.0)
table, _, _ = build_pow_lut(bits, bits, x_r, y_r)

xt = np.linspace(x_r[0], x_r[1], 1000)
yt = np.linspace(y_r[0], y_r[1], 1000)
XV, YV = np.meshgrid(xt, yt, indexing='ij')

Z_true = np.power(XV, YV)
# Z_approx = approx_pow_2d(XV, YV, table, x_r, y_r, bits, bits)
Z_approx = approx_pow_via_log_exp(XV, YV)
rel_error = np.abs(Z_approx - Z_true) / Z_true

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XV, YV, rel_error, cmap='viridis')
ax.set_title('Relative Error Surface ($x^y$)')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Rel Error')


plt.figure(figsize=(10, 8))
plt.contourf(XV, YV, np.log10(rel_error + 1e-15), levels=20, cmap='magma')
plt.colorbar(label='Log10(Relative Error)')
plt.title('Log Relative Error Map')
plt.xlabel('x')
plt.ylabel('y')
plt.show()