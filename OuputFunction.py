import numpy as np
import scipy.integrate as integrate
from astropy.cosmology import WMAP9 as cosmo
from astropy import constants
from astropy import units as u
import scipy.interpolate
import math


# Rescales data points within specified bins
def RescalingCenters(DataSet, Nbins, Bins_limit=None):

    # If Bins_limit is not provided, determine the limits from DataSet
    if Bins_limit == None:
        bins_max = DataSet[-1][0]
        bins_min = DataSet[0][0]
    else:
        bins_max = Bins_limit[1]
        bins_min = Bins_limit[0]
        
    x_data = DataSet[:, 0]
    bins_edges = np.logspace(np.log10(bins_min), np.log10(bins_max), Nbins + 1)
    bins_centers = np.array([0.5 * (bins_edges[i] + bins_edges[i + 1]) for i in range(Nbins)])
    y_data = DataSet[:, 1]

    y_stackable = np.zeros(Nbins)

    # Iterate through bins and rescale data points
    for j in range(Nbins):
        for i in range(len(x_data)):
            if (x_data[i] < bins_edges[j + 1]) and (x_data[i] > bins_edges[j]):
                x_data[i] = bins_centers[j]
                y_stackable[j] = round(y_data[i])
            else:
                pass
        
    return bins_edges, bins_centers, y_stackable

# Creates histogram from given data
def HistMaker(bar_edges, bar_centers, bar_Y):
    data = []
    for i in range(len(bar_edges) - 1):
        ii = 1
        while ii <= bar_Y[i]:
            data.append(np.log10(bar_centers[i]))
            ii += 1
    return data

"""
-------------------------------
"""

# Power Law distribution function
def PowerLaw(x, A, a):
    return A * np.exp(-x * a)

# Gaussian distribution function
def Gaussian(x, s, mu):
    return (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * s**2))

# Double Gaussian distribution function
def Double_Gaussian(x, s1, mu1, A1, s2, mu2, A2):
    return A1 * np.exp(-(x - mu1)**2 / (2 * s1**2)) + A2 * np.exp(-(x - mu2)**2 / (2 * s2**2))

"""
-------------------------------
"""

# Calculates the number of bins required for histogram plotting
def binnage(data):
    l = len(data)
    if l == 0:
        l = 1
    return int(2 * np.ceil(np.log2(l)) + 1)

# Prepares data for plotting
def output_prep(zs, ts, coal_time_datas, M_tot, dt, label_change, colors, ax, bx, cx, dx, ex, c_z, c_mtot):

    coal_time_datas = np.concatenate(coal_time_datas)

    coal_nbins = binnage(coal_time_datas)
    ts_nbins = binnage(ts)
    entries, edges, _ = ax.hist(ts, bins=np.logspace(np.log10(np.min(ts)), np.log10(np.max(ts)), ts_nbins), density=True, alpha=0.5, color=colors, label=label_change)

    zs_nbins = binnage(zs)
    z_hist, z_binedges = np.histogram(zs, zs_nbins, density=True)
    z_bins_centers = np.array([0.5 * (z_binedges[i] + z_binedges[i + 1]) for i in range(zs_nbins)])
    bx.bar(z_bins_centers, z_hist / c_z, width=np.diff(z_binedges), align='center', alpha=0.5, color=colors, label=label_change)

    entries, edges, _ = cx.hist(coal_time_datas, coal_nbins, density=True, alpha=0.5, color=colors, label=label_change)

    coal_time_unscaled = np.log10(10 ** np.array(coal_time_datas) - dt).tolist()
    entries, edges, _ = dx.hist(coal_time_unscaled, bins=np.logspace(np.log10(np.min(coal_time_unscaled)), np.log10(np.max(coal_time_unscaled)), coal_nbins), density=True, alpha=0.5, color=colors, label=label_change)

    m_tot = np.concatenate(M_tot)
    print(np.var(np.log10(m_tot)), np.mean(np.log10(m_tot)))
    M_tot_nbins = binnage(m_tot)
    M_tot_hist, M_tot_binedges = np.histogram(m_tot, np.logspace(np.log10(np.min(m_tot)), np.log10(np.max(m_tot)), M_tot_nbins), density=True)
    M_tot_centers = np.sqrt(M_tot_binedges[1:] * M_tot_binedges[:-1])
    ex.bar(M_tot_centers, M_tot_hist / c_mtot, width=np.diff(M_tot_binedges), align='center', alpha=0.5, color=colors, label=label_change)
    return
