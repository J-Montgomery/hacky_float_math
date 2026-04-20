import numpy as np
import matplotlib.pyplot as plt

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def approx_exp2_interpolated_2bit(x):
    exponent = np.floor(x).astype(int)
    fractional_part = x - exponent

    table = np.array([1.0, 1.18920712, 1.41421356, 1.68179283, 2.0], dtype=np.float32)

    scaled_f = fractional_part * 4.0
    idx = np.clip(np.floor(scaled_f).astype(int), 0, 3)
    frac = scaled_f - idx

    y0 = table[idx]
    y1 = table[idx + 1]

    exp2_f = y0 + frac * (y1 - y0)

    return np.ldexp(exp2_f, exponent)


def approx_log2_interpolated_2bit(x):
    mantissa, exponent = np.frexp(x)
    m = mantissa * 2.0
    e = (exponent - 1).astype(np.float32)

    table = np.array(
        [
            0.00000000,  # log2(1.00)
            0.32192809,  # log2(1.25)
            0.58496250,  # log2(1.50)
            0.80735492,  # log2(1.75)
            1.00000000,  # log2(2.00)
        ],
        dtype=np.float32,
    )

    f = m - 1.0
    scaled_f = f * 4.0

    idx = np.floor(scaled_f).astype(int)
    idx = np.clip(idx, 0, 3)

    fraction = scaled_f - idx

    y0 = table[idx]
    y1 = table[idx + 1]

    log2_m = y0 + fraction * (y1 - y0)

    return e + log2_m


def approx_exp(x):
    return approx_exp2_interpolated_2bit(x * LOG2E)


def approx_ln(x):
    return approx_log2_interpolated_2bit(x) * LN2


def approx_tanh(x):
    abs_x = np.abs(x)
    abs_x = np.clip(abs_x, 0, 40)
    e2x = approx_exp(2.0 * abs_x)
    res = 1.0 - 2.0 / (e2x + 1.0)
    return np.sign(x) * res


def approx_arctanh(x):
    # arctanh(x) = 0.5 * ln((1+x)/(1-x))
    x_clamped = np.clip(x, -0.999, 0.999)
    ratio = (1.0 + x_clamped) / (1.0 - x_clamped)
    return 0.5 * approx_ln(ratio)

x_tanh = np.linspace(-5, 5, 1000)
y_tanh_true = np.tanh(x_tanh)
y_tanh_approx = approx_tanh(x_tanh)
rel_err_tanh = (y_tanh_approx - y_tanh_true) / (np.abs(y_tanh_true) + 1e-9)

x_atanh = np.linspace(-0.95, 0.95, 1000)
y_atanh_true = np.arctanh(x_atanh)
y_atanh_approx = approx_arctanh(x_atanh)
rel_err_atanh = (y_atanh_approx - y_atanh_true) / (np.abs(y_atanh_true) + 1e-9)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_tanh, y_tanh_true, "b-", label="True tanh(x)", alpha=0.6)
plt.plot(x_tanh, y_tanh_approx, "r--", label="Approx tanh(x) (2-bit interp)")
plt.title("$\tanh(x)$ Approximation")
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_tanh, rel_err_tanh, "g-")
plt.title("Relative Error of $\tanh(x)$")
plt.xlabel("x")
plt.ylabel("Relative Error")
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_atanh, y_atanh_true, "b-", label="True arctanh(x)", alpha=0.6)
plt.plot(x_atanh, y_atanh_approx, "r--", label="Approx arctanh(x) (2-bit interp)")
plt.title("$\mathrm{arctanh}(x)$ Approximation")
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_atanh, rel_err_atanh, "g-")
plt.title("Relative Error of $\mathrm{arctanh}(x)$")
plt.xlabel("x")
plt.ylabel("Relative Error")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Relative error max for tanh:", np.max(np.abs(rel_err_tanh)))
print("Relative error max for arctanh:", np.max(np.abs(rel_err_atanh)))
