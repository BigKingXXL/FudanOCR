import numpy as np

from interfaces.q_utils import *

#-------------------------------------------------------------------------
# Distiller quantization function
#-------------------------------------------------------------------------
def distiller_quantize(x, num_bits, alpha):
    min_q_val, max_q_val = get_quantized_range(num_bits, signed=True)
    scale = symmetric_linear_quantization_scale_factor(num_bits, alpha)
    q = linear_quantize_clamp(x, scale, min_q_val, max_q_val)
    x = linear_dequantize(q, scale)
    return x

#-------------------------------------------------------------------------
# MSE for clip on histogram
#-------------------------------------------------------------------------
def mse_histogram_clip(bin_x, bin_y, num_bits, alpha):
   # Clipping error: sum over bins outside the clip threshold
   idx = np.abs(bin_x) > alpha
   mse = np.sum((np.abs(bin_x[idx]) - alpha)**2 * bin_y[idx])
   # Quantization error: sum over bins inside the clip threshold
   idx = np.abs(bin_x) <= alpha
   bin_xq = distiller_quantize(bin_x[idx], num_bits, alpha)
   mse += np.sum((bin_x[idx] - bin_xq)**2 * bin_y[idx])
   return mse

#-------------------------------------------------------------------------
# ACIQ method
#   ACIQ: Analytical Clipping for Integer Quantization of Neural Networks
#   https://arxiv.org/pdf/1810.05723.pdf
# Code taken and modified from:
#   https://github.com/submission2019/AnalyticalScaleForIntegerQuantization/blob/master/mse_analysis.py
#-------------------------------------------------------------------------
# 1. Find Gaussian and Laplacian clip thresholds
# 2. Estimate the MSE and choose the correct distribution
alpha_gauss   = {2:1.47818312, 3:1.80489289, 4:2.19227856, 5:2.57733584, 6:2.94451183, 7:3.29076248, 8:3.61691335,
                 9:4.21632968, 10:4.49417044, 11:4.75930917, 12:5.01321806, 13:5.25708493, 14:5.49196879, 15:5.71869998, 16:5.93797065} #added by gftm
alpha_laplace = {2:2.33152939, 3:3.04528770, 4:4.00378631, 5:5.08252088, 6:6.23211675, 7:7.42700429, 8:8.65265030,
                 9:11.16268502, 10: 12.44059133, 11: 13.72838476, 12:15.02446475, 13:16.32758309, 14:17.63674861, 15:18.95116231, 16:20.27017164} #added by gftm
gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)

def find_clip_aciq(values, num_bits):
    # Gaussian clip
    # This is how the ACIQ code calculates sigma
    sigma = ((np.max(values) - np.min(values)) * gaussian_const) / ((2 * np.log(values.size)) ** 0.5)
    #sigma = np.sqrt(np.sum((values - np.mean(values))**2) / (values.size-1))
    alpha_g = alpha_gauss[num_bits] * sigma
    # Laplacian clip
    b = np.mean(np.abs(values - np.mean(values)))
    alpha_l = alpha_laplace[num_bits] * b

    # Build histogram
    max_abs = np.max(np.abs(values))
    bin_range = (-max_abs, max_abs)
    bin_y, bin_edges = np.histogram(values, bins=101, range=bin_range,
                                    density=True)
    bin_x = 0.5*(bin_edges[:-1] + bin_edges[1:])

    # Pick the best fitting distribution
    mse_gauss = mse_histogram_clip(bin_x, bin_y, num_bits, alpha_g)
    mse_laplace = mse_histogram_clip(bin_x, bin_y, num_bits, alpha_l)

    alpha_best = alpha_g if mse_gauss < mse_laplace else alpha_l
    print(" ACIQ alpha_best = %7.4f / %7.4f" % (alpha_best, max_abs))
    return alpha_best
