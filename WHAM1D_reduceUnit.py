#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:42:12 2025

@author: wu
"""

import matplotlib.pyplot as plt
import statsmodels.api as statsm
import scipy.stats as stats
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm


class WHAMAnalyzer:
    def __init__(self, windows_data, centers, k_bias, T, num_bins):
        """
        Parameters:
        - windows_data: list of numpy arrays, each containing the 1D coordinate data sampled for a given window.
        - centers: list of floats, the target coordinate (bias center) for each window, corresponding to windows_data.
        - k_bias: float, the spring constant of the bias potential.
        - T: float, temperature.
        - num_bins: int, number of bins for the histogram (default is 50).
        - unit: reduce unit for the temperature.
        """
        self.windows_data = windows_data
        self.centers = np.array(centers)
        self.k_bias = k_bias
        self.T = T
        self.num_bins = num_bins
        self.beta = 1.0 / T # reduce unit

    @staticmethod
    def gaussian_overlap(mean1, std1, mean2, std2):
        """
        Calculate the overlap probability between two Gaussian distributions.
        Overlap = ∫ min{N(x; mean1, std1), N(x; mean2, std2)} dx
        """
        x_min = min(mean1 - 5*std1, mean2 - 5*std2)
        x_max = max(mean1 + 5*std1, mean2 + 5*std2)
        x_values = np.linspace(x_min, x_max, 1000)
        pdf1 = norm.pdf(x_values, mean1, std1)
        pdf2 = norm.pdf(x_values, mean2, std2)
        overlap = np.trapz(np.minimum(pdf1, pdf2), x_values)
        return overlap

    
    def wham_energy(self, tolerance=1e-6, max_iter=100000):
        """
        Use the WHAM method to reconstruct the free energy profile and calculate the activation energy (free energy barrier)
        from the umbrella sampling data.
        
        Returns:
        - activation_energy: The free energy barrier (difference between the maximum and minimum of F(x)).
        - bin_centers: The centers of the histogram bins.
        - F_x: The reconstructed free energy profile, F(x) = -k_BT ln(P(x)).
        """
        # Concatenate all window data to determine the global range
        all_data = np.concatenate(self.windows_data)
        xmin, xmax = all_data.min(), all_data.max()
        bins = np.linspace(xmin, xmax, self.num_bins + 1)
        dx = bins[1] - bins[0]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        M = len(self.windows_data)

        # Construct histograms for each window and record the total number of samples (N_i) for each window
        n_i_x = np.zeros((M, self.num_bins))
        N_i = np.zeros(M)
        for i in range(M):
            counts, _ = np.histogram(self.windows_data[i], bins=bins)
            n_i_x[i, :] = counts
            N_i[i] = counts.sum()

        # Initialize the free energy offsets f_i and the unbiased probability distribution P(x)
        f_i = np.zeros(M)
        P_x = np.ones(self.num_bins) / (self.num_bins * dx)
        omega = 0.5 * self.k_bias * (bin_centers[None, :] - self.centers[:, None])**2
        # WHAM iterative procedure
        for iteration in range(max_iter):
            denominator = np.zeros(self.num_bins)
            for i in range(M):
                # Calculate the bias potential for window i at each bin center:
                # U_i = 0.5 * self.k_bias * (bin_centers - self.centers[i])**2
                denominator += N_i[i] * np.exp(-self.beta * (omega[i] - f_i[i]))
            new_P_x = np.zeros(self.num_bins)
            for j in range(self.num_bins):
                numerator = np.sum(n_i_x[:, j])
                new_P_x[j] = numerator / (denominator[j])
            # Normalize P(x) so that its integral equals 1
            # print(np.sum(new_P_x))
            new_P_x = new_P_x / (np.sum(new_P_x) * dx)
            new_f_i = np.zeros(M)
            for i in range(M):
                # U_i = 0.5 * self.k_bias * (bin_centers - self.centers[i])**2
                new_f_i[i] = -np.log(np.sum(new_P_x * np.exp(-self.beta * omega[i])) * dx) / self.beta
            if np.max(np.abs(new_f_i - f_i)) < tolerance:
                f_i = new_f_i
                P_x = new_P_x
                print(f"WHAM converged after {iteration} iterations.")
                break
            f_i = new_f_i
            P_x = new_P_x
        else:
            print("WHAM did not converge within the maximum number of iterations.")

        # Calculate the free energy profile: F(x) = -k_BT ln(P(x))
        F_x = -1.0 / self.beta * np.log(P_x)
        # Define the activation energy as the difference between the maximum and minimum of F(x)
        energy_scale = F_x - F_x.min()

        return bin_centers, energy_scale, P_x

    def data_overlap_checking(self, threshold=0.1):
        """
        Check the overlap of the data distributions between windows.
        It is assumed that the data in each window approximately follows a Gaussian distribution.
        This method calculates the mean and standard deviation for each window, then computes the overlap area 
        between adjacent windows using the gaussian_overlap function, and compares it to the threshold (e.g., 0.1 means 10%).
        """
        window_stats = []
        for idx, window in enumerate(self.windows_data):
            mean_val = window.mean()
            std_val = window.std()
            window_stats.append((mean_val, std_val))    
            print(f"Window {idx}: mean = {mean_val:.3f}, std = {std_val:.3f}")
        print("\nOverlap analysis between adjacent windows:")
        for i in range(len(window_stats) - 1):
            mean1, std1 = window_stats[i]
            mean2, std2 = window_stats[i+1]
            overlap_area = self.gaussian_overlap(mean1, std1, mean2, std2)
            print(f"Overlap between Window {i} and Window {i+1}: {overlap_area*100:.2f}%")
            if overlap_area >= threshold:
                print(f"Windows {i} and {i+1} show sufficient overlap.")
            else:
                print(f"Windows {i} and {i+1} do NOT show sufficient overlap.")

    def plot_combined_histogram(self):
        """
        Plot the combined histogram of all windows' data.
        """
        all_data = np.concatenate(self.windows_data)
        xmin, xmax = all_data.min(), all_data.max()
        bins = np.linspace(xmin, xmax, self.num_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        counts, _ = np.histogram(all_data, bins=bins, density=True)
        
        
        plt.figure(figsize=(8,6))
        plt.plot(bin_centers, counts)
        plt.xlabel('Interparticle Distance')
        plt.ylabel('Frequency')
        plt.title('Combined Histogram of Windows Data')
        plt.show()
        
