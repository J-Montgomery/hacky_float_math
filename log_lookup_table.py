#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def build_log2_table(bits=4):
    n_entries = 1 << bits
    f_values = np.linspace(0, 1, n_entries, endpoint=False)
    table = np.log2(1 + f_values)
    return table

def approx_log2(x, table, bits=4):
    """
    x = 2^E * (1 + f)
    log2(x) = E + log2(1+f)
    """
    mantissa, exponent = np.frexp(x)
    m = mantissa * 2
    e = exponent - 1
    
    f = m - 1
    n_entries = 1 << bits
    index = (f * n_entries).astype(int)
    index = np.clip(index, 0, n_entries - 1)
    
    return e + table[index]

def build_log2_lut_interpolated(bits=4):
    n_entries = 1 << bits
    x_lut = np.linspace(1, 2, n_entries + 1)
    y_lut = np.log2(x_lut)
    
    table = y_lut[:-1]
    slopes = np.diff(y_lut)
    
    return table, slopes

def approx_log2_interpolated(x, table, slopes, bits=4):
    """
    x = 2^E * (1 + f)
    log2(x) = E + log2(1+f)
    """
    mantissa, exponent = np.frexp(x)
    m = mantissa * 2
    e = exponent - 1
    
    f = m - 1
    n_entries = 1 << bits
    
    scaled_f = f * n_entries
    index = np.floor(scaled_f).astype(int)
    index = np.clip(index, 0, n_entries - 1)
    fraction = scaled_f - index
    
    return e + table[index] + fraction * slopes[index]

def approx_log2_averaged(x, error_table, bits=4):
    x_float32 = np.asarray(x, dtype=np.float32)
    i = np.frombuffer(x_float32.tobytes(), dtype=np.int32)
    bit_log = (i - 0x3f800000) / 8388608.0
    
    mantissa, exponent = np.frexp(x)
    f = (mantissa * 2.0) - 1.0
    
    n_entries = 1 << bits
    scaled_f = f * n_entries
    idx0 = np.floor(scaled_f).astype(int)
    idx0 = np.clip(idx0, 0, n_entries - 1)
    frac = scaled_f - idx0
    err0 = error_table[idx0]
    
    # if idx + 1 is out of bounds, use 0.0 (since error at f=1.0 is 0)
    next_idx = idx0 + 1
    err1 = np.where(next_idx < n_entries, 
                    np.take(error_table, next_idx, mode='clip'), 
                    0.0)
    
    correction = err0 + frac * (err1 - err0)
    
    return bit_log + correction


def build_error_lut(bits=4):
    n_entries = 1 << bits
    f_values = np.linspace(0, 1, n_entries, endpoint=False)
    error_table = np.log2(1 + f_values) - f_values
    
    return error_table

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

bits = 1
table = build_log2_table(bits)
table2, slopes2 = build_log2_lut_interpolated(bits)
table3 = build_error_lut(bits)

x_vals = np.linspace(2, 8.0, 1000)
y_true = np.log2(x_vals)
y_approx = approx_log2(x_vals, table, bits)
y_interp = approx_log2_interpolated(x_vals, table2, slopes2, bits)
y_averaged = approx_log2_interpolated_2bit(x_vals)


plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_true, label='Exact $log_2(x)$', color='blue', alpha=0.5)
plt.plot(x_vals, y_approx, label=f'Approximation ({bits}-bit LUT)', color='red', linestyle='--')
plt.title(f'Log2 Approximation using a Significand Lookup Table ({bits} bits)')
plt.xlabel('x')
plt.ylabel('$log_2(x)$')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_true, label='Exact $log_2(x)$', color='blue', alpha=0.5)
plt.plot(x_vals, y_interp, label=f'Interpolated Approximation ({bits}-bit LUT)', color='red', linestyle='--')
plt.title(f'Log2 Approximation using a Significand Lookup Table ({bits} bits)')
plt.xlabel('x')
plt.ylabel('$log_2(x)$')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_true, label='Exact $log_2(x)$', color='blue', alpha=0.5)
plt.plot(x_vals, y_averaged, label=f'bitcast average ({bits}-bit LUT)', color='red', linestyle='--')
plt.title(f'Log2 Approximation using a bitcast + Significand Lookup Table ({bits} bits)')
plt.xlabel('x')
plt.ylabel('$log_2(x)$')
plt.legend()
plt.grid(True)

def build_exp2_table(bits=4):
    """Builds a lookup table for 2^f where f in [0, 1)."""
    n_entries = 1 << bits
    f_values = np.linspace(0, 1, n_entries, endpoint=False)
    table = 2**f_values
    return table

def approx_exp2(x, table, bits=4):
    """
    Approximates 2^x using the lookup table.
    x = I + f
    2^x = 2^I * 2^f
    """
    i = np.floor(x).astype(int)
    f = x - i
    
    n_entries = 1 << bits
    index = (f * n_entries).astype(int)
    index = np.clip(index, 0, n_entries - 1)
    
    return (2.0**i) * table[index]


table_exp = build_exp2_table(bits)
x_vals_exp = np.linspace(-2, 3, 1000)
y_true_exp = 2**x_vals_exp
y_approx_exp = approx_exp2(x_vals_exp, table_exp, bits)

plt.figure(figsize=(10, 6))
plt.plot(x_vals_exp, y_true_exp, label='Exact $2^x$', color='blue', alpha=0.5)
plt.plot(x_vals_exp, y_approx_exp, label=f'Approximation ({bits}-bit LUT)', color='green', linestyle='--')
plt.title(f'Exp2 ($2^x$) Approximation using a Fractional Lookup Table ({bits} bits)')
plt.xlabel('x')
plt.ylabel('$2^x$')
plt.legend()
plt.grid(True)

rel_err_exp = (y_approx_exp - y_true_exp) / y_true_exp 

plt.figure(figsize=(10, 6))
plt.plot(x_vals_exp, rel_err_exp, label='Relative Error (exp)', color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.title(f'Relative Error of Exp2 ($2^x$) Approximation ({bits}-bit LUT)')
plt.xlabel('x')
plt.ylabel('Relative Error $\\frac{approx - true}{true}$')
plt.legend()
plt.grid(True)
#plt.show()


rel_err_log = (y_approx - y_true) / y_true 
rel_err_log_interp = (y_interp - y_true) / y_true 
rel_err_log_bitcast = (y_averaged - y_true) / y_true

plt.figure(figsize=(10, 6))
#plt.plot(x_vals, rel_err_log, label='Relative Error (approx)', color='blue')
plt.plot(x_vals, rel_err_log_interp, label='Relative Error (interp)', color='red')
plt.plot(x_vals, rel_err_log_bitcast, label='Relative Error (bitcast)', color='yellow')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.title(f'Relative Error of log2  Approximation ({bits}-bit LUT)')
plt.xlabel('x')
plt.ylabel('Relative Error $\\frac{approx - true}{true}$')
plt.legend()
plt.grid(True)
plt.show()

# size_t significant_bits = 23;
# size_t lut_bits = 4;
# uint32_t bits = std::bit_cast<uint32_t>(x);
# uint32_t index = (bits >> (significand_bits - lut_bits)) & 0xF;
# float result = exponent_part + log_lut[index];