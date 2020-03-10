import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
import numpy as np
import math
import itertools
import random
import multiprocessing as mp
from time import time

canonical_IMF_assumption = "Kroupa"



data_Arrigoni2010_dynamical_mass = [9.24, 9.18, 9.25, 9.00, 9.36, 9.22, 10.30, 8.32, 10.61, 11.85, 11.89, 10.69, 10.48, 11.16, 10.85, 10.27, 11.41, 10.02, 10.19, 10.50, 9.45, 10.59, 9.34, 10.26, 10.85, 9.21, 10.12, 11.91, 10.89, 9.74, 10.72, 11.45, 9.88, 11.84, 11.09, 11.13, 10.26, 10.39, 10.56, 9.90, 10.50, 9.70, 9.34, 11.03, 11.07, 10.78, 10.73, 9.71, 10.08, 10.33, 9.56, 10.09, 11.40, 10.72, 10.68, 10.72, 11.21, 10.63, 11.38, 10.39, 11.25, 11.00, 10.68, 11.54, 10.20, 11.29, 11.58, 11.36, 11.19]
data_Arrigoni2010_dynamical_mass_error = [0.43, 0.39, 0.48, 0.45, 0.46, 0.50, 0.33, 0.28, 0.15, 0.35, 0.34, 0.30, 0.34, 0.27, 0.28, 0.42, 0.22, 0.31, 0.40, 0.31, 0.45, 0.31, 0.40, 0.32, 0.28, 0.40, 0.39, 0.21, 0.32, 0.45, 0.29, 0.39, 0.67, 0.36, 0.32, 0.38, 0.94, 0.35, 0.32, 0.28, 0.28, 0.44, 0.49, 0.35, 0.33, 0.28, 0.35, 0.42, 0.41, 0.45, 0.56, 0.38, 0.28, 0.23, 0.29, 0.33, 0.26, 0.31, 0.23, 0.36, 0.41, 0.30, 0.32, 0.38, 0.31, 0.35, 0.30, 0.37, 0.39]
data_Arrigoni2010_Z_H = [-0.431, -0.732, -0.412, -0.266, -0.904, 0.357, 0.049, -0.398, 0.281, 0.316, 0.359, 0.318, 0.160, 0.448, 0.210, -0.084, 0.260, -0.367, -0.079, -0.117, -0.313, 0.114, 0.068, -0.020, 0.157, -0.012, 0.104, 0.346, 0.187, -0.506, -0.062, 0.294, 0.155, 0.409, 0.335, 0.404, 0.226, 0.243, 0.384, 0.038, 0.159, 0.332, 0.029, 0.247, 0.285, 0.224, 0.361, 0.311, 0.153, 0.418, 0.074, 0.173, 0.284, 0.027, 0.185, 0.403, 0.100, 0.360, 0.296, 0.215, 0.266, 0.439, 0.180, 0.179, -0.128, 0.203, 0.320, 0.336, 0.266]
data_Arrigoni2010_Z_H_error_p = [0.083, 0.159, 0.098, 0.053, 0.220, 0.083, 0.068, 0.038, 0.068, 0.038, 0.068, 0.023, 0.053, 0.114, 0.038, 0.053, 0.053, 0.053, 0.053, 0.053, 0.068, 0.068, 0.053, 0.053, 0.098, 0.114, 0.053, 0.068, 0.068, 0.098, 0.023, 0.038, 0.083, 0.053, 0.023, 0.053, 0.068, 0.053, 0.053, 0.038, 0.038, 0.053, 0.083, 0.068, 0.068, 0.053, 0.038, 0.083, 0.098, 0.083, 0.068, 0.053, 0.038, 0.053, 0.053, 0.038, 0.053, 0.023, 0.038, 0.114, 0.038, 0.038, 0.053, 0.038, 0.053, 0.038, 0.023, 0.023, 0.038]
data_Arrigoni2010_Z_H_error_m = [0.098, 0.144, 0.098, 0.114, 0.189, 0.098, 0.083, 0.038, 0.038, 0.023, 0.053, 0.023, 0.053, 0.083, 0.023, 0.053, 0.038, 0.053, 0.053, 0.038, 0.083, 0.114, 0.068, 0.053, 0.068, 0.098, 0.038, 0.038, 0.038, 0.189, 0.023, 0.023, 0.053, 0.023, 0.023, 0.023, 0.053, 0.053, 0.053, 0.038, 0.023, 0.038, 0.083, 0.053, 0.068, 0.038, 0.023, 0.098, 0.083, 0.038, 0.083, 0.038, 0.023, 0.038, 0.023, 0.098, 0.038, 0.008, 0.023, 0.114, 0.023, 0.023, 0.023, 0.038, 0.038, 0.008, 0.023, 0.008, 0.008]
data_Arrigoni2010_Mg_Fe = [-0.005, -0.044, -0.076, 0.045, -0.164, 0.085, 0.162, -0.026, 0.153, 0.239, 0.179, 0.172, 0.093, 0.249, 0.137, 0.121, 0.109, 0.122, 0.173, 0.140, 0.111, 0.149, -0.045, 0.154, 0.127, -0.041, 0.092, 0.212, 0.128, 0.173, 0.104, 0.182, 0.134, 0.224, 0.114, 0.212, 0.096, 0.166, 0.095, 0.131, 0.168, 0.045, -0.011, 0.060, 0.098, 0.136, 0.186, 0.087, 0.088, 0.253, 0.109, 0.040, 0.175, 0.090, 0.162, 0.188, 0.197, 0.135, 0.215, 0.089, 0.217, 0.102, 0.110, 0.213, 0.031, 0.144, 0.173, 0.218, 0.151]
data_Arrigoni2010_Mg_Fe_error_p = [0.066, 0.126, 0.076, 0.045, 0.187, 0.045, 0.025, 0.025, 0.025, 0.015, 0.035, 0.015, 0.015, 0.056, 0.015, 0.035, 0.025, 0.035, 0.035, 0.025, 0.035, 0.025, 0.035, 0.025, 0.035, 0.056, 0.025, 0.025, 0.025, 0.045, 0.015, 0.015, 0.025, 0.025, 0.015, 0.015, 0.025, 0.025, 0.025, 0.025, 0.015, 0.025, 0.045, 0.035, 0.035, 0.015, 0.015, 0.045, 0.056, 0.025, 0.045, 0.015, 0.015, 0.025, 0.015, 0.005, 0.015, 0.015, 0.015, 0.056, 0.015, 0.015, 0.015, 0.015, 0.025, 0.015, 0.005, 0.005, 0.015]
data_Arrigoni2010_Mg_Fe_error_m = [0.045, 0.096, 0.056, 0.035, 0.500, 0.035, 0.015, 0.005, 0.005, 0.005, 0.015, 0.005, 0.015, 0.025, 0.005, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.025, 0.015, 0.015, 0.035, 0.005, 0.015, 0.005, 0.025, 0.005, 0.005, 0.015, 0.015, 0.005, 0.005, 0.015, 0.005, 0.025, 0.005, 0.005, 0.015, 0.035, 0.015, 0.025, 0.005, 0.005, 0.035, 0.035, 0.015, 0.035, 0.015, 0.005, 0.005, 0.005, 0.025, 0.005, 0.005, 0.005, 0.045, 0.005, 0.015, 0.005, 0.005, 0.015, 0.005, 0.005, 0.005, 0.005]

data_Arrigoni2010_logage = [0.807, 0.779, 0.707, 0.544, 0.603, 0.397, 1.126, 0.975, 0.926, 0.860, 0.601, 0.531, 0.762, 0.653, 0.950, 1.110, 0.510, 1.073, 1.146, 1.129, 1.010, 0.931, 0.323, 0.913, 1.008, 0.476, 0.923, 1.026, 0.994, 1.308, 0.981, 1.008, 0.864, 0.822, 0.519, 0.719, 0.810, 0.897, 0.574, 0.653, 0.969, 0.522, 0.632, 0.710, 0.720, 0.930, 0.763, 0.422, 0.542, 0.574, 0.464, 0.514, 1.136, 0.924, 0.946, 0.602, 1.127, 0.560, 0.980, 0.413, 0.981, 0.325, 0.790, 1.218, 0.692, 0.924, 1.051, 0.963, 0.911]
Universe_minus_Arrigoni2010_age = []
data_Arrigoni2010_logage_error_p = [0.114, 0.326, 0.129, 0.235, 0.159, 0.114, 0.098, 0.068, 0.129, 0.098, 0.129, 0.053, 0.098, 0.326, 0.068, 0.098, 0.038, 0.114, 0.098, 0.098, 0.083, 0.083, 0.023, 0.083, 0.114, 0.235, 0.068, 0.083, 0.068, 0.220, 0.038, 0.053, 0.098, 0.189, 0.038, 0.053, 0.159, 0.098, 0.159, 0.068, 0.038, 0.083, 0.098, 0.129, 0.189, 0.083, 0.144, 0.083, 0.159, 0.114, 0.023, 0.038, 0.053, 0.068, 0.068, 0.235, 0.098, 0.023, 0.068, 0.098, 0.053, 0.038, 0.083, 0.068, 0.053, 0.023, 0.053, 0.038, 0.068]
data_Arrigoni2010_logage_error_m = [0.250, 0.295, 0.235, 0.068, 0.189, 0.068, 0.114, 0.053, 0.159, 0.083, 0.098, 0.023, 0.068, 0.174, 0.038, 0.098, 0.053, 0.068, 0.129, 0.083, 0.053, 0.068, 0.023, 0.053, 0.083, 0.083, 0.038, 0.053, 0.053, 0.220, 0.023, 0.053, 0.098, 0.114, 0.008, 0.144, 0.174, 0.174, 0.068, 0.098, 0.023, 0.053, 0.159, 0.159, 0.174, 0.083, 0.068, 0.098, 0.083, 0.129, 0.008, 0.038, 0.053, 0.053, 0.038, 0.114, 0.083, 0.023, 0.038, 0.098, 0.038, 0.008, 0.068, 0.068, 0.144, 0.023, 0.023, 0.023, 0.038]
Universe_age_Planck_2015 = 13.8  # https://ui.adsabs.harvard.edu/abs/2016A%26A...594A..13P/abstract
Universe_minus_Arrigoni2010_age_other = []
data_Arrigoni2010_age = []
data_Arrigoni2010_age_error_p = []
data_Arrigoni2010_age_error_m = []
for i in range(len(data_Arrigoni2010_logage)):
    logage = data_Arrigoni2010_logage[i]
    age = 10**logage
    age_p = 10**(logage + data_Arrigoni2010_logage_error_p[i])
    age_m = 10**(logage - data_Arrigoni2010_logage_error_m[i])
    data_Arrigoni2010_age.append(age)
    data_Arrigoni2010_age_error_p.append(age_p - age)
    data_Arrigoni2010_age_error_m.append(age - age_m)
    Universe_minus_Arrigoni2010_age.append(Universe_age_Planck_2015-age)
age_limit = []
age_limit_p = []
age_limit_m = []
age_limit_upper = []
for i in range(len(data_Arrigoni2010_logage)):
    age_limit.append(Universe_minus_Arrigoni2010_age[i])
    age_limit_p.append(data_Arrigoni2010_age_error_m[i])
    age_limit_m.append(data_Arrigoni2010_age_error_p[i])
    age_limit_upper.append((Universe_minus_Arrigoni2010_age[i]+data_Arrigoni2010_age_error_m[i])*2)
age_limit_other = []
age_limit_p_other = []
age_limit_m_other = []
age_limit_upper_other = []
for i in range(len(data_Arrigoni2010_logage)):
    age_limit_other.append(data_Arrigoni2010_age[i])
    age_limit_p_other.append(data_Arrigoni2010_age_error_p[i])
    age_limit_m_other.append(data_Arrigoni2010_age_error_m[i])
    age_limit_upper_other.append((data_Arrigoni2010_age[i]+data_Arrigoni2010_age_error_p[i])*2)









def function_alpha_thomas(mass_thomas__):
    alpha_thomas__ = -0.459 + 0.062 * mass_thomas__
    return alpha_thomas__


def plot_obs(plot_data=True, plot_dirction=False):
    ######## Fig 1 ########## plot [Z/X] observations
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(1, figsize=(4, 3.5))

    plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
    plt.ylabel(r'[Z/X]')

    plt.plot(mass__list, mean_data_Z_H__list, c='k', zorder=7)
    plt.plot(mass__list, up_data_Z_H__list, c='k', ls='dashed', zorder=5)
    plt.plot(mass__list, down_data_Z_H__list, c='k', ls='dashed', zorder=5)
    # plt.fill_between(mass__list, down_data_Z_H__list, up_data_Z_H__list, alpha=0.4, facecolor='royalblue', linewidth=0)

    if plot_data == True:
        plt.errorbar(data_Arrigoni2010_dynamical_mass, data_Arrigoni2010_Z_H,
                     xerr=[data_Arrigoni2010_dynamical_mass_error, data_Arrigoni2010_dynamical_mass_error],
                     yerr=[data_Arrigoni2010_Z_H_error_m, data_Arrigoni2010_Z_H_error_p], capsize=1, elinewidth=0.3,
                     capthick=0.5, fmt='none', c='0.5')
        plt.scatter(data_Arrigoni2010_dynamical_mass, data_Arrigoni2010_Z_H, marker='x', c='0.5', zorder=6)


    if plot_dirction == True:
        if IMF == 'Kroupa':
            plt.plot(*zip(*plot_ZX_SFT), lw=0.7, c='b', zorder=8)
            plt.plot(*zip(*plot_ZX_STF), lw=0.7, c='g', zorder=8)
            plt.plot(*zip(*plot_ZX_SFR), lw=0.7, c='r', zorder=8)

            plt.scatter(*zip(*plot_ZX_SFT), c='b', zorder=9, s=20, label=r'$t_{\rm sf}$')
            plt.scatter(*zip(*plot_ZX_STF), c='g', zorder=9, s=20, label=r'$f_{\rm st}$')
            plt.scatter(*zip(*plot_ZX_SFR), c='r', zorder=9, s=20, label=r'log$_{10}(SFR)$')

            plt.scatter([plot_ZX_SFR[2][0]], [plot_ZX_SFR[2][1]], c='k', zorder=9, s=20)

            # plt.plot(*zip(*plot_ZX_SFT_igimf))
            # plt.plot(*zip(*plot_ZX_STF_igimf))
            # plt.plot(*zip(*plot_ZX_SFR_igimf))

            for a_point in plot_ZX_SFT_label:
                plt.annotate(round(a_point[2]) / 100, (a_point[0] + 0.03, a_point[1] + 0.03), fontsize=9, color='b',
                             zorder=10)
            for a_point in plot_ZX_STF_label:
                plt.annotate(a_point[2], (a_point[0] + 0.04, a_point[1] - 0.07), fontsize=9, color='g', zorder=10)
            for a_point in plot_ZX_SFR_label:
                plt.annotate(round(a_point[2]), (a_point[0] - 0.13, a_point[1] + 0.02), fontsize=9, color='r',
                             zorder=10)
            plt.legend(prop={'size': 8}, loc='lower right')
        elif IMF == 'igimf':
            plt.plot(*zip(*plot_ZX_SFT_igimf), lw=0.7, c='b', zorder=8)
            plt.plot(*zip(*plot_ZX_STF_igimf), lw=0.7, c='g', zorder=8)
            plt.plot(*zip(*plot_ZX_SFR_igimf), lw=0.7, c='r', zorder=8)

            plt.scatter(*zip(*plot_ZX_SFT_igimf), c='b', zorder=9, s=20, label=r'$t_{\rm sf}$')
            plt.scatter(*zip(*plot_ZX_STF_igimf), c='g', zorder=9, s=20, label=r'$f_{\rm st}$')
            plt.scatter(*zip(*plot_ZX_SFR_igimf), c='r', zorder=9, s=20, label=r'log$_{10}(SFR)$')

            plt.scatter([plot_ZX_SFR_igimf[2][0]], [plot_ZX_SFR_igimf[2][1]], c='k', zorder=9, s=20)

            # plt.plot(*zip(*plot_ZX_SFT_igimf))
            # plt.plot(*zip(*plot_ZX_STF_igimf))
            # plt.plot(*zip(*plot_ZX_SFR_igimf))

            for a_point in plot_ZX_SFT_label_igimf:
                plt.annotate(round(a_point[2]) / 100, (a_point[0] + 0.03, a_point[1] + 0.03), fontsize=9, color='b',
                             zorder=10)
            for a_point in plot_ZX_STF_label_igimf:
                plt.annotate(a_point[2], (a_point[0] + 0.04, a_point[1] - 0.07), fontsize=9, color='g', zorder=10)
            for a_point in plot_ZX_SFR_label_igimf:
                plt.annotate(round(a_point[2]), (a_point[0] - 0.13, a_point[1] + 0.02), fontsize=9, color='r',
                             zorder=10)
            plt.legend(prop={'size': 8}, loc='lower right')

    plt.xlim(9, 12)
    # plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig('ZX_obs.pdf', dpi=250)

    ######## Fig 2 ########## plot [Mg/Fe] observations
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(2, figsize=(4, 3.5))

    plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
    plt.ylabel(r'[Mg/Fe]')

    # plt.plot(mass__list, mean_data_MgFe__list, c='k', zorder=7)
    # plt.plot(mass__list, up_data_MgFe__list, c='k', ls='dotted', zorder=5)
    # plt.plot(mass__list, down_data_MgFe__list, c='k', ls='dotted', zorder=5)
    # plt.fill_between(mass__list, down_data_MgFe__list, up_data_MgFe__list, alpha=0.4, facecolor='chocolate', linewidth=0)


    mass_thomas = np.arange(6, 15, 1)
    alpha_thomas = function_alpha_thomas(mass_thomas)
    alpha_thomas_up = -0.459 + 0.062 * mass_thomas + 0.1
    alpha_thomas_low = -0.459 + 0.062 * mass_thomas - 0.1
    plt.plot(mass_thomas, alpha_thomas, c='k', zorder=7)
    plt.plot(mass_thomas, alpha_thomas_up, c='k', ls='dashed', zorder=5)
    plt.plot(mass_thomas, alpha_thomas_low, c='k', ls='dashed', zorder=5)
    # plt.fill_between(mass_thomas, alpha_thomas_low, alpha_thomas_up, alpha=0.4, facecolor='chocolate', linewidth=0)

    # if plot_data == True:
    #     plt.errorbar(data_Arrigoni2010_dynamical_mass, data_Arrigoni2010_Mg_Fe, xerr=[data_Arrigoni2010_dynamical_mass_error, data_Arrigoni2010_dynamical_mass_error],
    #                  yerr=[data_Arrigoni2010_Mg_Fe_error_m, data_Arrigoni2010_Mg_Fe_error_p], capsize=1, elinewidth=0.3, capthick=0.5, fmt='none', c='0.5')
    #
    #     plt.scatter(data_Arrigoni2010_dynamical_mass, data_Arrigoni2010_Mg_Fe, marker='x', c='chocolate', label='Arrigoni10 [Mg/Fe]', zorder=6)

    if plot_dirction == True:
        if IMF == 'igimf':
            plt.plot(*zip(*plot_MgFe_SFT_igimf), lw=0.7, c='b', zorder=8)
            plt.plot(*zip(*plot_MgFe_STF_igimf), lw=0.7, c='g', zorder=8)
            plt.plot(*zip(*plot_MgFe_SFR_igimf), lw=0.7, c='r', zorder=8)

            plt.scatter(*zip(*plot_MgFe_SFT_igimf), c='b', zorder=9, s=20, label=r'$t_{\rm sf}$')
            plt.scatter(*zip(*plot_MgFe_STF_igimf), c='g', zorder=9, s=20, label=r'$f_{\rm st}$')
            plt.scatter(*zip(*plot_MgFe_SFR_igimf), c='r', zorder=9, s=20, label=r'log$_{10}(SFR)$')

            plt.scatter([plot_MgFe_SFR_igimf[2][0]], [plot_MgFe_SFR_igimf[2][1]], c='k', zorder=9, s=20)

            # plt.plot(*zip(*plot_MgFe_SFT_igimf))
            # plt.plot(*zip(*plot_MgFe_STF_igimf))
            # plt.plot(*zip(*plot_MgFe_SFR_igimf))

            for a_point in plot_MgFe_SFT_label_igimf:
                plt.annotate(round(a_point[2]) / 100, (a_point[0] - 0.28, a_point[1] - 0.03), fontsize=9, color='b',
                             zorder=10)
            for a_point in plot_MgFe_STF_label_igimf:
                plt.annotate(a_point[2], (a_point[0] + 0.05, a_point[1] - 0.02), fontsize=9, color='g', zorder=10)
            for a_point in plot_MgFe_SFR_label_igimf:
                plt.annotate(round(a_point[2]), (a_point[0] - 0.14, a_point[1] + 0.005), fontsize=9, color='r',
                             zorder=10)
            plt.legend(prop={'size': 8}, loc='lower right')
        elif IMF == 'Kroupa':
            plt.plot(*zip(*plot_MgFe_SFT), lw=0.7, c='b', zorder=8)
            plt.plot(*zip(*plot_MgFe_STF), lw=0.7, c='g', zorder=8)
            plt.plot(*zip(*plot_MgFe_SFR), lw=0.7, c='r', zorder=8)

            plt.scatter(*zip(*plot_MgFe_SFT), c='b', zorder=9, s=20, label=r'$t_{\rm sf}$')
            plt.scatter(*zip(*plot_MgFe_STF), c='g', zorder=9, s=20, label=r'$f_{\rm st}$')
            plt.scatter(*zip(*plot_MgFe_SFR), c='r', zorder=9, s=20, label=r'log$_{10}(SFR)$')

            plt.scatter([plot_MgFe_SFR[2][0]], [plot_MgFe_SFR[2][1]], c='k', zorder=9, s=20)

            # plt.plot(*zip(*plot_MgFe_SFT_igimf))
            # plt.plot(*zip(*plot_MgFe_STF_igimf))
            # plt.plot(*zip(*plot_MgFe_SFR_igimf))

            for a_point in plot_MgFe_SFT_label:
                plt.annotate(round(a_point[2]) / 100, (a_point[0] - 0.28, a_point[1] - 0.03), fontsize=9, color='b',
                             zorder=10)
            for a_point in plot_MgFe_STF_label:
                plt.annotate(a_point[2], (a_point[0] + 0.05, a_point[1] - 0.02), fontsize=9, color='g', zorder=10)
            for a_point in plot_MgFe_SFR_label:
                plt.annotate(round(a_point[2]), (a_point[0] - 0.14, a_point[1] + 0.005), fontsize=9, color='r',
                             zorder=10)
            plt.legend(prop={'size': 8}, loc='lower right')

    plt.xlim(9, 12)
    plt.ylim(-0.2, 0.6)
    # plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig('MgFe_obs.pdf', dpi=250)
    return

mass_generate_from_Arrigoni2010 = []
Z_H_generate_from_Arrigoni2010 = []
MgFe_generate_from_Arrigoni2010 = []
j = 0
while j < 1000:
    i = 0
    length = len(data_Arrigoni2010_dynamical_mass)
    while i < length:
        mass = data_Arrigoni2010_dynamical_mass[i]
        sigma_mass = data_Arrigoni2010_dynamical_mass_error[i]
        random_mass_1 = random.normalvariate(mass, sigma_mass)
        random_mass_2 = random.normalvariate(mass, sigma_mass)
        Z_H = data_Arrigoni2010_Z_H[i]
        sigma_Z_H_p = data_Arrigoni2010_Z_H_error_p[i]
        sigma_Z_H_m = data_Arrigoni2010_Z_H_error_m[i]
        random_error_Z_H_p = abs(random.normalvariate(0, sigma_Z_H_p))
        random_error_Z_H_m = abs(random.normalvariate(0, sigma_Z_H_m))
        random_Z_H_1 = Z_H + random_error_Z_H_p
        random_Z_H_2 = Z_H - random_error_Z_H_m
        mass_generate_from_Arrigoni2010.append(random_mass_1)
        mass_generate_from_Arrigoni2010.append(random_mass_2)
        Z_H_generate_from_Arrigoni2010.append(random_Z_H_1)
        Z_H_generate_from_Arrigoni2010.append(random_Z_H_2)

        Mg_Fe = data_Arrigoni2010_Mg_Fe[i]
        sigma_MgFe_p = data_Arrigoni2010_Mg_Fe_error_p[i]
        sigma_MgFe_m = data_Arrigoni2010_Mg_Fe_error_m[i]
        random_error_MgFe_p = abs(random.normalvariate(0, sigma_MgFe_p))
        random_error_MgFe_m = abs(random.normalvariate(0, sigma_MgFe_m))
        random_MgFe_1 = Mg_Fe + random_error_MgFe_p
        random_MgFe_2 = Mg_Fe - random_error_MgFe_m
        MgFe_generate_from_Arrigoni2010.append(random_MgFe_1)
        MgFe_generate_from_Arrigoni2010.append(random_MgFe_2)
        (i) = (i+1)
    (j) = (j+1)
# plt.scatter(mass_generate_from_Arrigoni2010, Z_H_generate_from_Arrigoni2010, s=0.1)
# plt.scatter(mass_generate_from_Arrigoni2010, MgFe_generate_from_Arrigoni2010, s=0.1)

mass__list = []
mean_data_Z_H__list = []
up_data_Z_H__list = []
down_data_Z_H__list = []
mean_data_MgFe__list = []
up_data_MgFe__list = []
down_data_MgFe__list = []
# std_data_Z_H__list = []
# std_data_MgFe__list = []
mass__ = 8
while mass__ < 12.501:
    mass__upperlimit = mass__ + 0.5
    mass__lowerlimit = mass__ - 0.5
    data_mass__ = []
    data_Z_H__ = []
    data_MgFe__ = []
    i = 0
    length = len(mass_generate_from_Arrigoni2010)
    while i < length:
        if mass_generate_from_Arrigoni2010[i] < mass__upperlimit and mass_generate_from_Arrigoni2010[i] > mass__lowerlimit:
            data_Z_H__.append(Z_H_generate_from_Arrigoni2010[i])
            data_MgFe__.append(MgFe_generate_from_Arrigoni2010[i])
        (i) = (i + 1)
    mean_data_Z_H__ = np.mean(data_Z_H__)
    mean_data_MgFe__ = np.mean(data_MgFe__)
    std_data_Z_H__ = np.std(data_Z_H__)
    std_data_MgFe__ = np.std(data_MgFe__)
    mass__list.append(mass__)
    mean_data_Z_H__list.append(mean_data_Z_H__)
    up_data_Z_H__list.append(mean_data_Z_H__ + std_data_Z_H__)
    down_data_Z_H__list.append(mean_data_Z_H__ - std_data_Z_H__)
    mean_data_MgFe__list.append(mean_data_MgFe__)
    up_data_MgFe__list.append(mean_data_MgFe__+std_data_MgFe__)
    down_data_MgFe__list.append(mean_data_MgFe__-std_data_MgFe__)
    # std_data_Z_H__list.append(std_data_Z_H__)
    # std_data_MgFe__list.append(std_data_MgFe__)
    (mass__) = (mass__+0.1)


############## fit discrete relation with a function ##############

mass_obs_array = np.array(mass__list)
metal_obs_array_high = np.array(up_data_Z_H__list)
metal_obs_array = np.array(mean_data_Z_H__list)
metal_obs_array_low = np.array(down_data_Z_H__list)
fun_obs_metal_mass_high = interpolate.interp1d(mass_obs_array, metal_obs_array_high, kind='cubic')
fun_obs_metal_mass = interpolate.interp1d(mass_obs_array, metal_obs_array, kind='cubic')
fun_obs_metal_mass_low = interpolate.interp1d(mass_obs_array, metal_obs_array_low, kind='cubic')

alpha_obs_array_high = np.array(up_data_MgFe__list)
alpha_obs_array = np.array(mean_data_MgFe__list)
alpha_obs_array_low = np.array(down_data_MgFe__list)
# fun_obs_alpha_mass_high = interpolate.interp1d(mass_obs_array, alpha_obs_array_high, kind='cubic')
# fun_obs_alpha_mass = interpolate.interp1d(mass_obs_array, alpha_obs_array, kind='cubic')
# fun_obs_alpha_mass_low = interpolate.interp1d(mass_obs_array, alpha_obs_array_low, kind='cubic')



########################



SFEN_select = True

def impot_simulation(SFEN_select):
    plot_ZX_SFT = []
    plot_ZX_STF = []
    plot_ZX_SFR = []
    plot_MgFe_SFT = []
    plot_MgFe_STF = []
    plot_MgFe_SFR = []
    plot_ZX_SFT_label = []
    plot_ZX_STF_label = []
    plot_ZX_SFR_label = []
    plot_MgFe_SFT_label = []
    plot_MgFe_STF_label = []
    plot_MgFe_SFR_label = []
    plot_ZX_SFT_igimf = []
    plot_ZX_STF_igimf = []
    plot_ZX_SFR_igimf = []
    plot_MgFe_SFT_igimf = []
    plot_MgFe_STF_igimf = []
    plot_MgFe_SFR_igimf = []
    plot_ZX_SFT_label_igimf = []
    plot_ZX_STF_label_igimf = []
    plot_ZX_SFR_label_igimf = []
    plot_MgFe_SFT_label_igimf = []
    plot_MgFe_STF_label_igimf = []
    plot_MgFe_SFR_label_igimf = []
    # import simulation data points ###

    file = open('simulation_results_from_galaxy_evol/Metal_mass_relation.txt', 'r')
    data = file.readlines()
    file.close()

    file = open('simulation_results_from_galaxy_evol/Metal_mass_relation_igimf.txt', 'r')
    data_igimf = file.readlines()
    file.close()

    # plot simulation

    line = []
    line_igimf = []
    IMF = []
    IMF_igimf = []
    Log_SFR = []
    Log_SFR_igimf = []
    SFEN = []
    SFEN_igimf = []
    STF = []
    STF_igimf = []
    alive_stellar_mass = []
    alive_stellar_mass_igimf = []
    Dynamical_mass = []
    Dynamical_mass_igimf = []
    Mass_weighted_stellar_Mg_over_Fe = []
    Mass_weighted_stellar_Mg_over_Fe_igimf = []
    Mass_weighted_stellar_Z_over_X = []
    Mass_weighted_stellar_Z_over_X_igimf = []
    Mass_weighted_stellar_Fe_over_H_igimf = []
    luminosity_weighted_stellar_Mg_over_Fe = []
    luminosity_weighted_stellar_Mg_over_Fe_igimf = []
    luminosity_weighted_stellar_Z_over_X = []
    luminosity_weighted_stellar_Z_over_X_igimf = []
    gas_Mg_over_Fe = []
    gas_Mg_over_Fe_igimf = []
    gas_Z_over_X = []
    gas_Z_over_X_igimf = []
    middle_Mg_over_Fe = []
    middle_Mg_over_Fe_igimf = []
    middle_Z_over_X = []
    middle_Z_over_X_igimf = []
    error_Mg_over_Fe = []
    error_Mg_over_Fe_igimf = []
    error_Z_over_X = []
    error_Z_over_X_igimf = []
    AAA_igimf = []  # [Z/H] = [Fe/H] + AAA[α/Fe]
    error_if_AAA_is_1 = []
    points = np.array([[0, 0, 0]])  # [Dynamical_mass, Mass_weighted_stellar_Mg_over_Fe, Mass_weighted_stellar_Z_over_X]
    points_igimf = np.array([[0, 0, 0]])
    metal_values = np.array([[0]])  #
    metal_values_igimf = np.array([[0]])  #
    alpha_values = np.array([[0]])  #
    alpha_values_igimf = np.array([[0]])  #

    length_data = len(data)
    i = 1
    while i < length_data:
        line_i = [float(x) for x in data[i].split()]
        if (canonical_IMF_assumption == "Kroupa" and line_i[0] == 0 and
                (SFEN_select is True or line_i[2] == SFEN_select)) \
                or (canonical_IMF_assumption == "Salpeter" and line_i[0] == 2):
            if True:  # line_i[2] == 100:
                line.append(line_i)
                IMF.append(line_i[0])
                Log_SFR.append(line_i[1])
                SFEN.append(line_i[2])  # these are not used for interpolation, "points" are.
                STF.append(line_i[3])  # these are not used for interpolation, "points" are.
                alive_stellar_mass.append(line_i[4])
                Dynamical_mass.append(line_i[5])  # these are not used for interpolation, "points" are.
                Mass_weighted_stellar_Mg_over_Fe.append(line_i[6])
                Mass_weighted_stellar_Z_over_X.append(line_i[7])
                points = np.append(points, [[line_i[2], line_i[3], line_i[5]]], axis=0)  ##### +1
                metal_values = np.append(metal_values, [[line_i[7]]], axis=0)
                alpha_values = np.append(alpha_values, [[line_i[6]]], axis=0)
                # metal_values = np.append(metal_values, [[(line_i[7] + line_i[9]) / 2]], axis=0)
                # alpha_values = np.append(alpha_values, [[(line_i[6] + line_i[8]) / 2]], axis=0)
                if len(line_i) > 8:
                    gas_Mg_over_Fe.append(line_i[8])
                    gas_Z_over_X.append(line_i[9])
                    middle_Mg_over_Fe.append((line_i[6] + line_i[8]) / 2)
                    middle_Z_over_X.append((line_i[7] + line_i[9]) / 2)
                    error_Mg_over_Fe.append(abs((line_i[6] - line_i[8]) / 2))
                    error_Z_over_X.append(abs((line_i[7] - line_i[9]) / 2))
                else:
                    gas_Mg_over_Fe.append(None)
                    gas_Z_over_X.append(None)
                if len(line_i) > 10:
                    luminosity_weighted_stellar_Mg_over_Fe.append(line_i[10])
                    luminosity_weighted_stellar_Z_over_X.append(line_i[11])
                else:
                    luminosity_weighted_stellar_Mg_over_Fe.append(None)
                    luminosity_weighted_stellar_Z_over_X.append(None)
                if line_i[1] == 3 and line_i[3] == 0.9 and (line_i[2] == 50 or line_i[2] == 10 or line_i[2] == 25):
                    plot_ZX_SFT.append([line_i[5], line_i[7]])
                    plot_MgFe_SFT.append([line_i[5], line_i[6] - error_alpha])
                    plot_ZX_SFT_label.append([line_i[5], line_i[7], line_i[2]])
                    plot_MgFe_SFT_label.append([line_i[5], line_i[6] - error_alpha, line_i[2]])
                if line_i[1] == 3 and line_i[2] == 25 and (line_i[3] == 0.5 or line_i[3] == 0.9 or line_i[3] == 1.3):
                    plot_ZX_STF.append([line_i[5], line_i[7]])
                    plot_MgFe_STF.append([line_i[5], line_i[6] - error_alpha])
                    plot_ZX_STF_label.append([line_i[5], line_i[7], line_i[3]])
                    plot_MgFe_STF_label.append([line_i[5], line_i[6] - error_alpha, line_i[3]])
                if line_i[2] == 25 and line_i[3] == 0.9 and (line_i[1] == 3 or line_i[1] == 2 or line_i[1] == 1):
                    plot_ZX_SFR.append([line_i[5], line_i[7]])
                    plot_MgFe_SFR.append([line_i[5], line_i[6] - error_alpha])
                    plot_ZX_SFR_label.append([line_i[5], line_i[7], line_i[1]])
                    plot_MgFe_SFR_label.append([line_i[5], line_i[6] - error_alpha, line_i[1]])
        (i) = (i + 1)
    plot_ZX_SFT = sorted(plot_ZX_SFT, key=lambda l: l[0])
    plot_MgFe_SFT = sorted(plot_MgFe_SFT, key=lambda l: l[0])
    plot_ZX_STF = sorted(plot_ZX_STF, key=lambda l: l[0])
    plot_MgFe_STF = sorted(plot_MgFe_STF, key=lambda l: l[0])
    plot_ZX_SFR = sorted(plot_ZX_SFR, key=lambda l: l[0])
    plot_MgFe_SFR = sorted(plot_MgFe_SFR, key=lambda l: l[0])

    length_data_igimf = len(data_igimf)
    i = 1
    while i < length_data_igimf:
        line_i = [float(x) for x in data_igimf[i].split()]
        if SFEN_select is True or line_i[2] == SFEN_select:
            if STF_select is True or line_i[3] == STF_select:
                line_igimf.append(line_i)
                IMF_igimf.append(line_i[0])
                Log_SFR_igimf.append(line_i[1])
                SFEN_igimf.append(line_i[2])  # these are not used for interpolation, "points" are.
                STF_igimf.append(line_i[3])  # these are not used for interpolation, "points" are.
                alive_stellar_mass_igimf.append(line_i[4])
                Dynamical_mass_igimf.append(line_i[5])  # these are not used for interpolation, "points" are.
                Mass_weighted_stellar_Mg_over_Fe_igimf.append(line_i[6])
                Mass_weighted_stellar_Z_over_X_igimf.append(line_i[7])
                Mass_weighted_stellar_Fe_over_H_igimf.append(line_i[12])
                AAA_igimf.append((line_i[7]-line_i[12])/line_i[6])  # [Z/H] = [Fe/H] + AAA[α/Fe]
                error_if_AAA_is_1.append(line_i[7]-line_i[12]-line_i[6])  # [Z/H] - [Fe/H] - [α/Fe]
                points_igimf = np.append(points_igimf, [[line_i[2], line_i[3], line_i[5]]], axis=0)
                metal_values_igimf = np.append(metal_values_igimf, [[line_i[7]]], axis=0)
                alpha_values_igimf = np.append(alpha_values_igimf, [[line_i[6]]], axis=0)
                # metal_values_igimf = np.append(metal_values_igimf, [[(line_i[7] + line_i[9]) / 2]], axis=0)
                # alpha_values_igimf = np.append(alpha_values_igimf, [[(line_i[6] + line_i[8]) / 2]], axis=0)
                # gas_Mg_over_Fe_igimf.append(line_i[8])
                # gas_Z_over_X_igimf.append(line_i[9])
                # middle_Mg_over_Fe_igimf.append((line_i[6] + line_i[8]) / 2)
                # middle_Z_over_X_igimf.append((line_i[7] + line_i[9]) / 2)
                # error_Mg_over_Fe_igimf.append(abs((line_i[6] - line_i[8]) / 2))
                # error_Z_over_X_igimf.append(abs((line_i[7] - line_i[9]) / 2))
                if len(line_i) > 10:
                    luminosity_weighted_stellar_Mg_over_Fe_igimf.append(line_i[10])
                    luminosity_weighted_stellar_Z_over_X_igimf.append(line_i[11])
                else:
                    luminosity_weighted_stellar_Mg_over_Fe_igimf.append(None)
                    luminosity_weighted_stellar_Z_over_X_igimf.append(None)
                    if line_i[1] == 0 and line_i[3] == 0.3 and (line_i[2] == 10 or line_i[2] == 20 or line_i[2] == 50):
                        plot_ZX_SFT_igimf.append([line_i[5], line_i[7]])
                        plot_MgFe_SFT_igimf.append([line_i[5], line_i[6] - error_alpha])
                        plot_ZX_SFT_label_igimf.append([line_i[5], line_i[7], line_i[2]])
                        plot_MgFe_SFT_label_igimf.append([line_i[5], line_i[6] - error_alpha, line_i[2]])
                    if line_i[1] == 0 and line_i[2] == 20 and (line_i[3] == 0.1 or line_i[3] == 0.3 or line_i[3] == 0.5):
                        plot_ZX_STF_igimf.append([line_i[5], line_i[7]])
                        plot_MgFe_STF_igimf.append([line_i[5], line_i[6] - error_alpha])
                        plot_ZX_STF_label_igimf.append([line_i[5], line_i[7], line_i[3]])
                        plot_MgFe_STF_label_igimf.append([line_i[5], line_i[6] - error_alpha, line_i[3]])
                    if line_i[2] == 20 and line_i[3] == 0.3 and (line_i[1] == -2 or line_i[1] == 0 or line_i[1] == 2):
                        plot_ZX_SFR_igimf.append([line_i[5], line_i[7]])
                        plot_MgFe_SFR_igimf.append([line_i[5], line_i[6] - error_alpha])
                        plot_ZX_SFR_label_igimf.append([line_i[5], line_i[7], line_i[1]])
                        plot_MgFe_SFR_label_igimf.append([line_i[5], line_i[6] - error_alpha, line_i[1]])
        (i) = (i + 1)
    plot_ZX_SFT_igimf = sorted(plot_ZX_SFT_igimf, key=lambda l: l[0])
    plot_MgFe_SFT_igimf = sorted(plot_MgFe_SFT_igimf, key=lambda l: l[0])
    plot_ZX_STF_igimf = sorted(plot_ZX_STF_igimf, key=lambda l: l[0])
    plot_MgFe_STF_igimf = sorted(plot_MgFe_STF_igimf, key=lambda l: l[0])
    plot_ZX_SFR_igimf = sorted(plot_ZX_SFR_igimf, key=lambda l: l[0])
    plot_MgFe_SFR_igimf = sorted(plot_MgFe_SFR_igimf, key=lambda l: l[0])

    points = np.delete(points, 0, axis=0)
    points_igimf = np.delete(points_igimf, 0, axis=0)
    metal_values = np.delete(metal_values, 0, axis=0)
    metal_values_igimf = np.delete(metal_values_igimf, 0, axis=0)
    alpha_values = np.delete(alpha_values, 0, axis=0)
    alpha_values_igimf = np.delete(alpha_values_igimf, 0, axis=0)
    return points, points_igimf, metal_values, metal_values_igimf, alpha_values, alpha_values_igimf, \
           Dynamical_mass_igimf, Mass_weighted_stellar_Z_over_X_igimf, Mass_weighted_stellar_Mg_over_Fe_igimf, \
           Dynamical_mass, Mass_weighted_stellar_Z_over_X, Mass_weighted_stellar_Mg_over_Fe, \
           plot_ZX_SFT, plot_ZX_STF, plot_ZX_SFR, plot_MgFe_SFT, plot_MgFe_STF, plot_MgFe_SFR, \
           plot_ZX_SFT_label, plot_ZX_STF_label, plot_ZX_SFR_label, plot_MgFe_SFT_label, plot_MgFe_STF_label, \
           plot_MgFe_SFR_label, STF, STF_igimf, SFEN, SFEN_igimf, Mass_weighted_stellar_Fe_over_H_igimf, \
           plot_ZX_SFT_igimf, plot_MgFe_SFT_igimf, plot_ZX_STF_igimf, plot_MgFe_STF_igimf, plot_ZX_SFR_igimf, \
           plot_MgFe_SFR_igimf, plot_ZX_SFT_label_igimf, plot_ZX_STF_label_igimf, plot_ZX_SFR_label_igimf, \
           plot_MgFe_SFT_label_igimf, plot_MgFe_STF_label_igimf, plot_MgFe_SFR_label_igimf, \
           AAA_igimf, error_if_AAA_is_1, Log_SFR_igimf


#################### setup interpolation ####################

def interpolate_simulation(points, points_igimf, metal_values, metal_values_igimf, alpha_values, alpha_values_igimf,
                           grid_SFEN, grid_STF, grid_Dynamical_mass):
    grid_alpha1 = griddata(points, alpha_values, (grid_SFEN, grid_STF, grid_Dynamical_mass), method='linear')
    grid_metal1 = griddata(points, metal_values, (grid_SFEN, grid_STF, grid_Dynamical_mass), method='linear')
    grid_alpha_igimf1 = griddata(points_igimf, alpha_values_igimf, (grid_SFEN, grid_STF, grid_Dynamical_mass),
                                 method='linear')
    grid_metal_igimf1 = griddata(points_igimf, metal_values_igimf, (grid_SFEN, grid_STF, grid_Dynamical_mass),
                                 method='linear')
    return grid_alpha1, grid_metal1, grid_alpha_igimf1, grid_metal_igimf1


### plot simulated results ###
def plot_simulation(IMF, Dynamical_mass, Dynamical_mass_igimf, Mass_weighted_stellar_Z_over_X,
                    Mass_weighted_stellar_Z_over_X_igimf, Mass_weighted_stellar_Mg_over_Fe,
                    Mass_weighted_stellar_Mg_over_Fe_igimf, STF, STF_igimf, SFEN, SFEN_igimf,
                    Mass_weighted_stellar_Fe_over_H_igimf, plot_ZX_SFT_igimf, plot_MgFe_SFT_igimf, plot_ZX_STF_igimf,
                    plot_MgFe_STF_igimf, plot_ZX_SFR_igimf, plot_MgFe_SFR_igimf, plot_ZX_SFT_label_igimf,
                    plot_ZX_STF_label_igimf, plot_ZX_SFR_label_igimf, plot_MgFe_SFT_label_igimf,
                    plot_MgFe_STF_label_igimf, plot_MgFe_SFR_label_igimf, AAA_igimf, error_if_AAA_is_1, Log_SFR_igimf):
    # fig, ax = plt.subplots()
    # ax.hist(AAA_igimf, bins=200)  # AAA = 1.27+-0.35
    # print(AAA_igimf)
    # ax.hist(error_if_AAA_is_1, bins=20)

    fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=4, ncols=1, sharex=True)

    if IMF == "igimf":
        ax1.scatter(Dynamical_mass_igimf, Mass_weighted_stellar_Z_over_X_igimf, s=3)
        # ax1.set_xlabel(r'log $M_{\rm gal}$')
        ax1.set_ylabel('[Z/X]')
        ax2.scatter(Dynamical_mass_igimf, Mass_weighted_stellar_Mg_over_Fe_igimf, s=3)
        # ax2.set_xlabel(r'log $M_{\rm gal}$')
        ax2.set_ylabel('[Mg/Fe]')
        ax3.scatter(Dynamical_mass_igimf, Mass_weighted_stellar_Fe_over_H_igimf, s=3)
        ax3.set_ylabel('[Fe/H]')
        # ax4.scatter(Dynamical_mass_igimf, AAA_igimf, s=3, label='[Fe/H] igimf')
        ax4.scatter(Dynamical_mass_igimf, error_if_AAA_is_1, s=3)
        ax4.set_ylabel('[Z/H]-[Fe/H]-[α/Fe]')
        ax4.set_xlabel(r'log$_{10}(M_{\rm gal})$')
        plt.subplots_adjust(hspace=.0)
        plt.tight_layout()

        # the error of AAA is determined by the TIgwIMF, that is, SFR and metallciity evolution of the galaxy.
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        fig = plt.figure(5, figsize=(4, 3.5))
        # plt.xlim(-0.5, 2.2)
        # plt.ylim(0.23, 0.6)
        plt.scatter(Log_SFR_igimf, error_if_AAA_is_1, s=3)
        # plt.legend(prop={'size': 6}, loc='best')
        plt.xlabel(r'log$_{10}(SFR)$')
        plt.ylabel('[Z/H]-[Fe/H]-[α/Fe]')
        plt.tight_layout()


        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        fig = plt.figure(6, figsize=(4, 3.5))
        plt.hist(AAA_igimf, bins=200)  # AAA = 1.27+-0.35
        plt.xlabel(r'A = ([Z/H]-[Fe/H])/[α/Fe]')
        plt.ylabel('#')
        # print(AAA_igimf)
        # ax.hist(error_if_AAA_is_1, bins=20)
        plt.tight_layout()


        # for i, txt in enumerate(SFEN_igimf):
        #     ax.annotate(txt, (Dynamical_mass_igimf[i], Mass_weighted_stellar_Mg_over_Fe_igimf[i]))
        #     ax.annotate(txt, (Dynamical_mass_igimf[i], Mass_weighted_stellar_Z_over_X_igimf[i]))
        # plt.errorbar(Dynamical_mass_igimf, middle_Z_over_X_igimf, yerr=error_Z_over_X_igimf, capsize=4, linestyle="None", fmt='o', label='[Z/X] igimf')
        # plt.errorbar(Dynamical_mass_igimf, middle_Mg_over_Fe_igimf, yerr=error_Mg_over_Fe_igimf, capsize=4, linestyle="None", fmt='o', label='[Mg/Fe] igimf')
    # elif IMF == "Kroupa":
    #     ax.scatter(Dynamical_mass, Mass_weighted_stellar_Z_over_X, s=3, label='[Z/X] Kroupa-IMF')
    #     ax.scatter(Dynamical_mass, Mass_weighted_stellar_Mg_over_Fe, s=3, label='[Mg/Fe] Kroupa-IMF')
    #     for i, txt in enumerate(STF):
    #         ax.annotate(txt, (Dynamical_mass[i], Mass_weighted_stellar_Mg_over_Fe[i]))
    #         # ax.annotate(txt, (Dynamical_mass[i], Mass_weighted_stellar_Z_over_X[i]))
    #     # plt.errorbar(Dynamical_mass, middle_Z_over_X, yerr=error_Z_over_X, capsize=4, linestyle="None", fmt='o', label='[Z/X] Kroupa-IMF')
    #     # plt.errorbar(Dynamical_mass, middle_Mg_over_Fe, yerr=error_Mg_over_Fe, capsize=4, linestyle="None", fmt='o', label='[Mg/Fe] Kroupa-IMF')

    # ### plot the [Mg/Fe] as a function of STF, labeled with metallicity.
    # fig, ax = plt.subplots()
    # modified_STF_igimf = []
    # for i in range(len(STF_igimf)):
    #     # modified_STF_igimf.append(STF_igimf[i]+SFEN_igimf[i]**(0.1)/(400**(0.1))/10+Dynamical_mass_igimf[i]/150)
    #     modified_STF_igimf.append(STF_igimf[i]+SFEN_igimf[i]+Dynamical_mass_igimf[i])
    # if IMF == "igimf":
    #     # ax.scatter(STF_igimf, Mass_weighted_stellar_Z_over_X_igimf, s=3, label='[Z/X] igimf')
    #     ax.scatter(modified_STF_igimf, Mass_weighted_stellar_Mg_over_Fe_igimf, s=3, label='[Mg/Fe] igimf', c=Mass_weighted_stellar_Z_over_X_igimf)
    #     for i, txt in enumerate(Mass_weighted_stellar_Z_over_X_igimf):
    #         ax.annotate(round(txt, 2), (modified_STF_igimf[i], Mass_weighted_stellar_Mg_over_Fe_igimf[i]))
    #         # ax.annotate(txt, (STF_igimf[i], Mass_weighted_stellar_Z_over_X_igimf[i]))
    # elif IMF == "Kroupa":
    #     ax.scatter(STF, Mass_weighted_stellar_Z_over_X, s=3, label='[Z/X] Kroupa-IMF')
    #     ax.scatter(STF, Mass_weighted_stellar_Mg_over_Fe, s=3, label='[Mg/Fe] Kroupa-IMF')
    #     for i, txt in enumerate(Mass_weighted_stellar_Z_over_X):
    #         ax.annotate(txt, (STF[i], Mass_weighted_stellar_Mg_over_Fe[i]))
    #         ax.annotate(txt, (STF[i], Mass_weighted_stellar_Z_over_X[i]))
    return


######### define functions that calculate likelihood #########


def calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha):
    if STF_j < len(grid_SFEN[SFEN_i]):
        SFEN_interpolate = grid_SFEN[SFEN_i][STF_j][Dynamical_mass_k]
        STF_interpolate = grid_STF[SFEN_i][STF_j][Dynamical_mass_k]
        Dynamical_mass_interpolate = grid_Dynamical_mass[SFEN_i][STF_j][Dynamical_mass_k]

        alpha_likelihood = calculate_likelihood_alpha(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_alpha)
        metal_likelihood = calculate_likelihood_metal(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal)

        total_likelihood = alpha_likelihood * metal_likelihood  ################################################
        # print(alpha_likelihood, metal_likelihood, SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
        return SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood
    else:
        return 0, 0, 0, 0

def analytical_Kroupa_grid(SFEN_i, STF_j):
    global number_of_SFEN, number_of_STF # 5-400, 0.1-1.5
    SFEN = 5 + (400-5)/number_of_SFEN*SFEN_i
    STF = 0.1 + 1.4/number_of_STF*STF_j
    log_SFEN = math.log(SFEN, 10)
    alpha_interpolate = 0.11*log_SFEN*STF - 0.38*STF -0.39*log_SFEN + 1.1
    return alpha_interpolate

def calculate_likelihood_alpha(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_alpha):
    Dynamical_mass_interpolate__ = grid_Dynamical_mass[SFEN_i][STF_j][Dynamical_mass_k]

    if IMF == "igimf":
        alpha_interpolate__ = grid_alpha_igimf1[SFEN_i][STF_j][Dynamical_mass_k]
    if IMF == "Kroupa":
        alpha_interpolate__ = grid_alpha1[SFEN_i][STF_j][Dynamical_mass_k]
        # alpha_interpolate__ = analytical_Kroupa_grid(SFEN_i, STF_j)

    # alpha_interpolate_obs__ = fun_obs_alpha_mass(Dynamical_mass_interpolate__) + error_alpha
    # std_alpha_obs = abs(
    #     fun_obs_alpha_mass_high(Dynamical_mass_interpolate__) - fun_obs_alpha_mass_low(Dynamical_mass_interpolate__))/2

    alpha_interpolate_obs__ = -0.459+0.062*Dynamical_mass_interpolate__ + error_alpha
    std_alpha_obs = 0.1

    mismatch_alpha = abs(alpha_interpolate__ - alpha_interpolate_obs__)
    x_alpha = mismatch_alpha / std_alpha_obs
    alpha_likelihood = math.erfc(x_alpha/2**0.5)

    return alpha_likelihood


def calculate_likelihood_metal(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal):
    Dynamical_mass_interpolate = grid_Dynamical_mass[SFEN_i][STF_j][Dynamical_mass_k]

    if IMF == "igimf":
        metal_interpolate = grid_metal_igimf1[SFEN_i][STF_j][Dynamical_mass_k]
    if IMF == "Kroupa":
        metal_interpolate = grid_metal1[SFEN_i][STF_j][Dynamical_mass_k]

    metal_interpolate_obs = fun_obs_metal_mass(Dynamical_mass_interpolate) + error_metal

    std_metal_obs = abs(
        fun_obs_metal_mass_high(Dynamical_mass_interpolate) - fun_obs_metal_mass_low(Dynamical_mass_interpolate))/2
    # std_metal_obs = std_alpha_obs
    mismatch_metal = abs(metal_interpolate - metal_interpolate_obs)
    x_metal = mismatch_metal / std_metal_obs
    metal_likelihood = math.erfc(x_metal/2**0.5)
    return metal_likelihood


######### SFT--galaxy-mass relations given by Thomas05 and Recchi09 #########

def f1(mass):
    '''Thomas05 equaiton 5'''
    SFT = math.exp(3.67-0.37*mass)
    return SFT
    # return math.exp(3.954-0.372*t)
def f2(mass):
    '''Recchi09'''
    SFT = math.exp(2.38-0.24*mass)
    return SFT
def f3(mass):
    '''de La Rosa 2011'''
    alpha = function_alpha_thomas(mass)
    SFT = (-15.3*alpha + 5.2)
    return SFT
def f4(mass):
    '''Richard McDermid 2015 equaiton 3'''
    alpha = function_alpha_thomas(mass)
    SFT = math.exp((0.28-alpha)/0.19)
    return SFT
t = [8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]  # galaxy mass
y1 = [f1(8.5), f1(9), f1(9.5), f1(10), f1(10.5), f1(11), f1(11.5), f1(12)]  # SFT
y2 = [f2(8.5), f2(9), f2(9.5), f2(10), f2(10.5), f2(11), f2(11.5), f2(12)]
y3 = [f3(8.5), f3(9), f3(9.5), f3(10), f3(10.5), f3(11), f3(11.5), f3(12)]
t4 = [9.5, 10, 10.5, 11, 11.5, 12]
y4 = [f4(9.5), f4(10), f4(10.5), f4(11), f4(11.5), f4(12)]



### find best likelihood and plot ###


############# single plots #############
def plot_likelihood():
    # STF_j = 42  ####################### 14 21 28 35 42
    STF_j = 42
    print(grid_STF_list[STF_j])
    # STF fixe for entire map:
    fig = plt.figure(31, figsize=(4, 3.5))
    plt.title(r"$f_s$$_t$ = {}".format(grid_STF_list[STF_j]))
    plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
    plt.ylabel(r'$t_{sf}$ [Gyr]')
    plt.xlim(9, 12)
    plt.ylim(0, 4)
    SFEN_interpolate_list = []
    STF_interpolate_list = []
    Dynamical_mass_interpolate_list = []
    total_likelihood_list = []
    best_mass_likelihood_list = []
    for Dynamical_mass_k in range(number_of_Dynamical_mass):
        # STF_j = round((0.33 - 0.23 / number_of_Dynamical_mass * Dynamical_mass_k) * number_of_STF)
        best_likelihood = 0
        for SFEN_i in range(number_of_SFEN):
            SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood = \
                calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
            SFEN_interpolate_list.append(SFEN_interpolate / 100)
            STF_interpolate_list.append(STF_interpolate)
            Dynamical_mass_interpolate_list.append(Dynamical_mass_interpolate)
            total_likelihood_list.append(total_likelihood)
            if total_likelihood > best_likelihood:
                best_likelihood = total_likelihood
        if best_likelihood == 0:
            best_mass_likelihood_list.append([None, None])
        else:
            best_mass_likelihood_list.append([Dynamical_mass_interpolate, best_likelihood*4])
        # print(STF_interpolate)
    Dynamical_mass_interpolate_list = [0, 0] + Dynamical_mass_interpolate_list
    SFEN_interpolate_list = [0, 0] + SFEN_interpolate_list
    total_likelihood_list = [0, 1] + total_likelihood_list
    sc = plt.scatter(Dynamical_mass_interpolate_list, SFEN_interpolate_list, c=total_likelihood_list, s=30)
    plt.colorbar(sc)
    ### plots for previous studies ###
    plt.plot(t, y1, c='b', lw=3, ls='dotted', label='Thomas05')
    # plt.plot(t, y3, c='g', lw=2, ls='dashed', label='deLaRosa11')
    # plt.plot(t4, y4, c='r', lw=2, label='McDermid15')
    mass_thomas_list = np.arange(9, 12.1, 0.1)
    SFT_thomas_list = []
    for i in mass_thomas_list:
        # alpha_interpolate_obs = fun_obs_alpha_mass(i)
        alpha_interpolate_obs = -0.459 + 0.062 * i + error_alpha
        SFT_thomas = math.exp((1 / 5 - alpha_interpolate_obs) * 6)
        SFT_thomas_list.append(SFT_thomas)
    best_mass_likelihood_list = sorted(best_mass_likelihood_list, key=lambda l: l[0])
    plt.plot(*zip(*best_mass_likelihood_list), c='darkorange', lw=1)
    # plt.plot(mass_thomas_list, SFT_thomas_list, c='k', ls='dotted', label='Thomas05-corrected')
    # plt.plot(t, y2, c='k', ls='-.', label='Recchi09')
    # plt.legend(prop={'size': 7}, loc='upper right')
    plt.tight_layout()
    plt.savefig('reproduce_1.pdf', dpi=250)
    return


def plot_Update():  # change line 480
    # STF fixe for entire map:
    fig = plt.figure(32, figsize=(4, 3.5))
    plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
    plt.ylabel(r'$t_{sf}$ [Gyr]')
    plt.xlim(9, 12)
    plt.ylim(0, 4)
    SFEN_interpolate_list = []
    STF_interpolate_list = []
    Dynamical_mass_interpolate_list = []
    total_likelihood_list = []
    best_mass_likelihood_list = []
    for Dynamical_mass_k in range(number_of_Dynamical_mass):
        STF_j = 46  #######################
        # STF_j = round((0.33 - 0.23 / number_of_Dynamical_mass * Dynamical_mass_k) * number_of_STF)
        best_likelihood = 0
        for SFEN_i in range(number_of_SFEN):
            SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood = \
                calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
            SFEN_interpolate_list.append(SFEN_interpolate / 100)
            STF_interpolate_list.append(STF_interpolate)
            Dynamical_mass_interpolate_list.append(Dynamical_mass_interpolate)
            total_likelihood_list.append(total_likelihood)
            if total_likelihood > best_likelihood:
                best_likelihood = total_likelihood
        if best_likelihood == 0:
            best_mass_likelihood_list.append([None, None])
        else:
            best_mass_likelihood_list.append([Dynamical_mass_interpolate, best_likelihood*4])
        # print(STF_interpolate)
    Dynamical_mass_interpolate_list = [0, 0] + Dynamical_mass_interpolate_list
    SFEN_interpolate_list = [0, 0] + SFEN_interpolate_list
    total_likelihood_list = [0, 1] + total_likelihood_list
    sc = plt.scatter(Dynamical_mass_interpolate_list, SFEN_interpolate_list, c=total_likelihood_list, s=30)
    plt.colorbar(sc)
    ### plots for previous studies ###
    plt.plot(t, y1, c='b', lw=3, ls='dotted', label='Thomas05')
    mass_thomas_list = np.arange(9, 12.1, 0.1)
    SFT_thomas_list = []
    for i in mass_thomas_list:
        # alpha_interpolate_obs = fun_obs_alpha_mass(i)
        alpha_interpolate_obs = -0.459 + 0.062 * i + error_alpha
        SFT_thomas = math.exp((1 / 5 - alpha_interpolate_obs) * 6)
        SFT_thomas_list.append(SFT_thomas)
    best_mass_likelihood_list = sorted(best_mass_likelihood_list, key=lambda l: l[0])
    plt.plot(*zip(*best_mass_likelihood_list), c='darkorange', lw=1)
    # plt.plot(mass_thomas_list, SFT_thomas_list, c='k', ls='dotted', label='Thomas05-corrected')
    # plt.plot(t, y2, c='k', ls='-.', label='Recchi09')
    # plt.legend(prop={'size': 7}, loc='upper right')
    plt.tight_layout()
    plt.savefig('Update_1.pdf', dpi=250)
    return




# Instead of fit both obseravtion simutanously,
# here first fit only Z to determine STF
# then fit Mg/Fe to determin SFT
# In this way, it mimics the Recchi or Thomas's work that only consider [Mg/Fe]
########################
###### Fig 2 & 6 #######


def calculate_for_a_galaxy_mass(Dynamical_mass_k):
    global error_metal, error_alpha
    print(Dynamical_mass_k + 1, '%')
    total_likelihood__ = 0
    SFEN_interpolate_list = []
    STF_interpolate_list = []
    Dynamical_mass_interpolate_list = []
    total_likelihood_list = []
    best_mass_likelihood_list = []  # [(mass1, SFEN1), (mass2, SFEN2)...]
    ### first (fit metal abundance to) determine the best STF for each galaxy-mass ###
    for STF_j__, SFEN_i__ in itertools.product(range(number_of_STF), range(number_of_SFEN)):
        # total_likelihood = calculate_likelihood_metal(SFEN_i__, STF_j__, Dynamical_mass_k, IMF, error_metal)
        total_likelihood = calculate_likelihood(SFEN_i__, STF_j__, Dynamical_mass_k, IMF, error_metal, error_alpha)[3]
        if not np.isnan(total_likelihood):
            if total_likelihood > total_likelihood__:
                total_likelihood__ = total_likelihood
                STF_j = STF_j__
    ### cut the plot where simulation grid is not available ###
    # SFEN_test_1 = round(number_of_SFEN / 10 * 8)
    # SFEN_test_2 = round(number_of_SFEN / 10 * 2)
    # total_likelihood_test_1 = calculate_likelihood(SFEN_test_1, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)[
    #     3]
    # total_likelihood_test_2 = calculate_likelihood(SFEN_test_2, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)[
    #     3]
    total_likelihood_test_1 = True
    total_likelihood_test_2 = True
    ### for each galaxy-mass with fixed STF calculate the likelihood (consider both metal and Mg/Fe) for each SFT ###
    if not (np.isnan(total_likelihood_test_1) and np.isnan(total_likelihood_test_2)):
        best_likelihood = 0
        for SFEN_i in range(number_of_SFEN):
            SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood = \
                calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
            SFEN_interpolate_list.append(SFEN_interpolate / 100)
            STF_interpolate_list.append(STF_interpolate)
            Dynamical_mass_interpolate_list.append(Dynamical_mass_interpolate)
            total_likelihood_list.append(total_likelihood)
            if not np.isnan(total_likelihood):
                if total_likelihood > best_likelihood:
                    best_likelihood = total_likelihood  #SFEN_interpolate
        if best_likelihood == 0:
            best_mass_likelihood_list.append([None, None])
        else:
            best_mass_likelihood_list.append([Dynamical_mass_interpolate, best_likelihood * 4])
    result = [SFEN_interpolate_list, STF_interpolate_list, Dynamical_mass_interpolate_list, total_likelihood_list,
              best_mass_likelihood_list]
    return result

def calculate_for_a_galaxy_mass_free_STF(Dynamical_mass_k):
    global error_metal, error_alpha
    print(Dynamical_mass_k + 1, '%')
    SFEN_interpolate_list = []
    STF_interpolate_list = []
    Dynamical_mass_interpolate_list = []
    total_likelihood_list = []
    best_mass_likelihood_list = []  # [(mass1, SFEN1), (mass2, SFEN2)...]

    best_likelihood = 0
    for SFEN_i in range(number_of_SFEN):
        total_likelihood__ = 0
        for STF_j__ in range(number_of_STF):
            total_likelihood = calculate_likelihood(SFEN_i, STF_j__, Dynamical_mass_k, IMF, error_metal, error_alpha)[3]
            if not np.isnan(total_likelihood):
                if total_likelihood > total_likelihood__:
                    total_likelihood__ = total_likelihood
                    STF_j = STF_j__
            else:
                STF_j = 0
        SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood = \
            calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
        SFEN_interpolate_list.append(SFEN_interpolate / 100)
        STF_interpolate_list.append(STF_interpolate)
        Dynamical_mass_interpolate_list.append(Dynamical_mass_interpolate)
        total_likelihood_list.append(total_likelihood)
        if not np.isnan(total_likelihood):
            if total_likelihood > best_likelihood:
                best_likelihood = total_likelihood  # SFEN_interpolate
    if best_likelihood == 0:
        best_mass_likelihood_list.append([None, None])
    else:
        best_mass_likelihood_list.append([Dynamical_mass_interpolate, best_likelihood * 4])
    result = [SFEN_interpolate_list, STF_interpolate_list, Dynamical_mass_interpolate_list, total_likelihood_list,
              best_mass_likelihood_list]
    return result


def plot_fig_likelihood():
    plot_number = 2
    global error_metal, error_alpha
    for error_metal__, error_alpha__ in itertools.product(error_metal_list, error_alpha_list):
        plt.figure(plot_number, figsize=(4, 3.5))
        plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
        plt.ylabel(r'$t_{sf}$ [Gyr]')
        plt.xlim(9, 12)
        plt.ylim(0, 4)
        error_metal = error_metal__
        error_alpha = error_alpha__
        print(error_metal, error_alpha)
        SFEN_interpolate_list = []
        STF_interpolate_list = []
        Dynamical_mass_interpolate_list = []
        total_likelihood_list = []
        best_mass_likelihood_list = []  # [(mass1, SFEN1), (mass2, SFEN2)...]
        prior_likelihood = 1  #math.erfc(abs(error_metal) / standard_deviation_systematic_metal / 2 ** 0.5) * \
                           # math.erfc(abs(error_alpha) / standard_deviation_systematic_alpha / 2 ** 0.5)
        # print("prior_likelihood:", prior_likelihood)
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(calculate_for_a_galaxy_mass_free_STF, #calculate_for_a_galaxy_mass,
                           [Dynamical_mass_k for Dynamical_mass_k in range(number_of_Dynamical_mass)])
        for i in range(number_of_Dynamical_mass):
            # print(results[i])
            SFEN_interpolate_list.extend(results[i][0])
            STF_interpolate_list.extend(results[i][1])
            Dynamical_mass_interpolate_list.extend(results[i][2])
            total_likelihood_list.extend(results[i][3])
            best_mass_likelihood_list.extend(results[i][4])
        pool.close()
        best_mass_likelihood_list_sorted = sorted(best_mass_likelihood_list, key=lambda l: l[0])

        # print("Plotting...")
        Dynamical_mass_interpolate_list = [0, 0] + Dynamical_mass_interpolate_list
        SFEN_interpolate_list = [0, 0] + SFEN_interpolate_list
        total_likelihood_list = [0, 1] + total_likelihood_list
        sc = plt.scatter(Dynamical_mass_interpolate_list, SFEN_interpolate_list, c=total_likelihood_list, s=10) #, cmap='gist_rainbow')
        plt.plot(*zip(*best_mass_likelihood_list_sorted), c='darkorange', lw=1)
        clb = plt.colorbar(sc)
        # clb.set_label('likelihood')
        # plt.title('[Z/X]_lit+({}); [Mg/Fe]_lit+({})'.format(error_metal, error_alpha), fontsize=8)

        ### plots for previous studies ###
        plt.plot(t, y1, c='b', lw=3, ls='dotted', label='Thomas05')
        plt.plot(t, y3, c='r', lw=2, ls='dashed', label='deLaRosa11')
        plt.plot([9, 12], [0.4, 0.4], c='r', lw=2, ls='dashed', label='Johansson12')
        plt.plot(t4, y4, c='r', lw=2, ls='dashdot', label='McDermid15')
        mass_thomas_list = np.arange(9, 12.1, 0.1)
        SFT_thomas_list = []
        for i in mass_thomas_list:
            # alpha_interpolate_obs = fun_obs_alpha_mass(i)
            alpha_interpolate_obs = -0.459 + 0.062 * i + error_alpha
            SFT_thomas = math.exp((1 / 5 - alpha_interpolate_obs) * 6)
            SFT_thomas_list.append(SFT_thomas)
        # plt.plot(mass_thomas_list, SFT_thomas_list, c='k', ls='dotted', label='Thomas05-corrected')
        # plt.plot(t, y2, c='k', ls='-.', label='Recchi09')
        # plt.legend(prop={'size': 7}, loc='upper right')

        plt.tight_layout()
        if IMF == "Kroupa":
            plt.savefig('Update_2_{}.pdf'.format(plot_number), dpi=250)
            # plot_number=2 is saved as Update_2_2.pdf for Fig5
            # plot_number=3 is saved as Update_2_3.pdf for Fig7.
        if IMF == "igimf":
            plt.savefig('likelihood_igimf_{}.pdf'.format(plot_number), dpi=250)
        plot_number += 1
    return





def plot_age_limit():
    from scipy import stats
    import numpy as np
    plt.figure(99, figsize=(4, 3.5))
    plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
    plt.ylabel(r'$t_{sf}$ [Gyr]')
    plt.xlim(9, 12)
    # plt.ylim(-1, 15)
    ### plots for previous studies ###
    plt.plot(t, y1, c='b', ls='dotted', label='Thomas05')
    plt.plot(t, y3, c='g', ls='dashed', label='deLaRosa11')
    plt.plot(t4, y4, c='r', label='McDermid15')
    mass_thomas_list = np.arange(9, 12.1, 0.1)
    SFT_thomas_list = []
    for i in mass_thomas_list:
        # alpha_interpolate_obs = fun_obs_alpha_mass(i)
        alpha_interpolate_obs = -0.459 + 0.062 * i + error_alpha
        SFT_thomas = math.exp((1 / 5 - alpha_interpolate_obs) * 6)
        SFT_thomas_list.append(SFT_thomas)
    # plt.plot(mass_thomas_list, SFT_thomas_list, c='k', ls='dotted', label='Thomas05-corrected')
    # plt.plot(t, y2, c='k', ls='-.', label='Recchi09')
    # plt.legend(prop={'size': 7}, loc='upper right')

    # # # plot age limitations:
    plt.scatter(data_Arrigoni2010_dynamical_mass, age_limit_upper, marker=11, c='darkorange', zorder=6)
    plt.scatter(data_Arrigoni2010_dynamical_mass, age_limit_upper_other, marker='x', c='r', zorder=6)
    # plt.errorbar(data_Arrigoni2010_dynamical_mass, age_limit,
    #              xerr=[data_Arrigoni2010_dynamical_mass_error, data_Arrigoni2010_dynamical_mass_error],
    #              yerr=[age_limit_m, age_limit_p], capsize=1, elinewidth=0.3,
    #              capthick=0.5, fmt='none', c='0.5')
    # plt.scatter(data_Arrigoni2010_dynamical_mass, age_limit, marker='x', c='r', zorder=6)
    # # plt.plot(data_Arrigoni2010_dynamical_mass, age_limit, c='r', lw=0.5, zorder=6)

    mass_generate = []
    age_generate = []
    j = 0
    while j < 3:
        i = 0
        length = len(data_Arrigoni2010_dynamical_mass)
        while i < length:
            mass = data_Arrigoni2010_dynamical_mass[i]
            sigma_mass = data_Arrigoni2010_dynamical_mass_error[i]
            random_mass_1 = random.normalvariate(mass, sigma_mass)
            age = age_limit_upper[i]
            random_age_1 = abs(random.normalvariate(0, age))
            mass_generate.append(random_mass_1)
            age_generate.append(random_age_1)
            (i) = (i + 1)
        (j) = (j + 1)
    mass_generate_other = []
    age_generate_other = []
    j = 0
    while j < 3:
        i = 0
        length = len(data_Arrigoni2010_dynamical_mass)
        while i < length:
            mass = data_Arrigoni2010_dynamical_mass[i]
            sigma_mass = data_Arrigoni2010_dynamical_mass_error[i]
            random_mass_1 = random.normalvariate(mass, sigma_mass)
            age = age_limit_upper_other[i]
            random_age_1 = abs(random.normalvariate(0, age))
            mass_generate_other.append(random_mass_1)
            age_generate_other.append(random_age_1)
            (i) = (i + 1)
        (j) = (j + 1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(mass_generate, age_generate)
    slope_other, intercept_other, r_value_other, p_value_other, std_err_other = \
        stats.linregress(mass_generate_other, age_generate_other)
    xx = np.linspace(9,12,2)
    yy = slope*xx + intercept
    plt.plot(xx, yy, c='darkorange')
    yy = slope_other*xx + intercept_other
    plt.plot(xx, yy, c='r')

    plt.scatter(mass_generate, age_generate)
    plt.scatter(mass_generate_other, age_generate_other)
    plt.tight_layout()
    plt.savefig('age_limit.pdf', dpi=250)
    return


### plot STF--mass... relation ###



def parallel_mass(Dynamical_mass_k):
    print(Dynamical_mass_k+1, '%')
    prior_likelihood = 1 #math.erfc(abs(error_metal) / standard_deviation_systematic_metal / 2 ** 0.5) * \
                       # math.erfc(abs(error_alpha) / standard_deviation_systematic_alpha / 2 ** 0.5)
    SFEN_interpolate_list = []
    STF_interpolate_list = []
    total_likelihood_list = []
    best_mass_SFEN_list = []
    best_STF_mass_list = []
    best_alpha_SFEN_list = []
    ### find the best-fit STF--galaxy-mass relation ###
    total_likelihood__ = 0
    # STF_j = 33
    for STF_j__, SFEN_i__ in itertools.product(range(number_of_STF), range(number_of_SFEN)):
        # total_likelihood = calculate_likelihood_metal(SFEN_i__, STF_j__, Dynamical_mass_k, IMF, error_metal)
        total_likelihood = calculate_likelihood(SFEN_i__, STF_j__, Dynamical_mass_k, IMF, error_metal, error_alpha)[3]
        if not np.isnan(total_likelihood):
            if total_likelihood > total_likelihood__:
                total_likelihood__ = total_likelihood
                STF_j = STF_j__
    if STF_j == 0:
        print("Warning: no solution for STF_j?")

    # ### for a given STF--galaxy-mass relation ###
    # # STF_j = round((0.1 + 0.9 / number_of_Dynamical_mass * Dynamical_mass_k) * number_of_STF)
    # STF_j = round((0.5 - 0 / number_of_Dynamical_mass * Dynamical_mass_k) * number_of_STF)

    # SFEN_test_1 = round(number_of_SFEN/10*8)
    # SFEN_test_2 = round(number_of_SFEN/10*2)
    # total_likelihood_test_1 = calculate_likelihood(SFEN_test_1, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)[3]
    # total_likelihood_test_2 = calculate_likelihood(SFEN_test_2, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)[3]
    total_likelihood_test_1 = True
    total_likelihood_test_2 = True
    if not (np.isnan(total_likelihood_test_1) and np.isnan(total_likelihood_test_2)):
        total_likelihood__ = 0
        SFEN__ = 0
        Dynamical_mass_interpolate = 0
        for SFEN_i in range(number_of_SFEN):
            SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood = \
                calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
            SFEN_interpolate_list.append(SFEN_interpolate / 100)
            total_likelihood_list.append(total_likelihood * prior_likelihood)
            if not np.isnan(total_likelihood):
                if total_likelihood > total_likelihood__:
                    total_likelihood__ = total_likelihood
                    SFEN__ = SFEN_interpolate
        if SFEN__ == 0 or Dynamical_mass_interpolate == 0:
            best_mass_SFEN_list.append([best_mass_SFEN_list[-1][0], best_mass_SFEN_list[-1][1]])
            best_alpha_SFEN_list.append([best_alpha_SFEN_list[-1][0], best_alpha_SFEN_list[-1][1]])
            best_STF_mass_list.append([best_STF_mass_list[-1][0], best_STF_mass_list[-1][1]])
        else:
            best_STF_mass_list.append([Dynamical_mass_interpolate, STF_interpolate])
            best_mass_SFEN_list.append([Dynamical_mass_interpolate, SFEN__ / 100])
            # alpha_interpolate_obs = fun_obs_alpha_mass(Dynamical_mass_interpolate) + error_alpha
            alpha_interpolate_obs = -0.459+0.062*Dynamical_mass_interpolate + error_alpha
            best_alpha_SFEN_list.append([SFEN__ / 100, alpha_interpolate_obs])
    result = [SFEN_interpolate_list, STF_interpolate_list, total_likelihood_list, best_STF_mass_list,
              best_mass_SFEN_list, best_alpha_SFEN_list]
    return result



def parallel_mass_free_STF(Dynamical_mass_k):
    print(Dynamical_mass_k+1, '%')
    prior_likelihood = 1
    SFEN_interpolate_list = []
    STF_interpolate_list = []
    total_likelihood_list = []
    best_mass_SFEN_list = []
    best_STF_mass_list = []
    best_alpha_SFEN_list = []
    best_likelihood = 0
    SFEN__ = 0
    Dynamical_mass_interpolate = 0
    for SFEN_i in range(number_of_SFEN):
        total_likelihood__ = 0
        for STF_j__ in range(number_of_STF):
            total_likelihood = calculate_likelihood(SFEN_i, STF_j__, Dynamical_mass_k, IMF, error_metal, error_alpha)[3]
            if not np.isnan(total_likelihood):
                if total_likelihood > total_likelihood__:
                    total_likelihood__ = total_likelihood
                    STF_j = STF_j__
            else:
                STF_j = 0
        SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood = \
            calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
        SFEN_interpolate_list.append(SFEN_interpolate / 100)
        total_likelihood_list.append(total_likelihood * prior_likelihood)
        if not np.isnan(total_likelihood):
            if total_likelihood > best_likelihood:
                best_likelihood = total_likelihood
                SFEN__ = SFEN_interpolate
    if not (SFEN__ == 0 or Dynamical_mass_interpolate == 0):
        # if len(best_mass_SFEN_list) > 0:
        #     best_mass_SFEN_list.append([best_mass_SFEN_list[-1][0], best_mass_SFEN_list[-1][1]])
        #     best_alpha_SFEN_list.append([best_alpha_SFEN_list[-1][0], best_alpha_SFEN_list[-1][1]])
        #     best_STF_mass_list.append([best_STF_mass_list[-1][0], best_STF_mass_list[-1][1]])
    # else:
        best_STF_mass_list.append([Dynamical_mass_interpolate, STF_interpolate])
        best_mass_SFEN_list.append([Dynamical_mass_interpolate, SFEN__ / 100])
        # alpha_interpolate_obs = fun_obs_alpha_mass(Dynamical_mass_interpolate) + error_alpha
        alpha_interpolate_obs = -0.459 + 0.062 * Dynamical_mass_interpolate + error_alpha
        best_alpha_SFEN_list.append([SFEN__ / 100, alpha_interpolate_obs])
    result = [SFEN_interpolate_list, STF_interpolate_list, total_likelihood_list, best_STF_mass_list,
              best_mass_SFEN_list, best_alpha_SFEN_list]
    return result


def plot_best_fits():
    print(error_metal, error_alpha)

    SFEN_interpolate_list = []
    STF_interpolate_list = []
    total_likelihood_list = []
    best_mass_SFEN_list = []
    best_STF_mass_list = []
    best_alpha_SFEN_list = []
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(parallel_mass_free_STF, [Dynamical_mass_k for Dynamical_mass_k in range(number_of_Dynamical_mass)])
    for i in range(number_of_Dynamical_mass):
        SFEN_interpolate_list.extend(results[i][0])
        STF_interpolate_list.extend(results[i][1])
        total_likelihood_list.extend(results[i][2])
        best_STF_mass_list.extend(results[i][3])
        best_mass_SFEN_list.extend(results[i][4])
        best_alpha_SFEN_list.extend(results[i][5])
    pool.close()
    best_mass_SFEN_list_sorted = sorted(best_mass_SFEN_list, key=lambda l: l[0])
    best_alpha_SFEN_list_sorted = sorted(best_alpha_SFEN_list, key=lambda l: l[0])
    best_STF_mass_list_sorted = sorted(best_STF_mass_list, key=lambda l: l[0])

    fig = plt.figure(1001, figsize=(4, 2.2))
    plt.plot(*zip(*best_STF_mass_list_sorted), c='k')
    plt.plot([9, 12], [0.5, 0.5], ls='dashed', c='k')
    plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
    plt.ylabel(r'$f_{\mathrm{st}}$')
    plt.xlim(9, 12)
    plt.ylim(0.45, 1.55)
    plt.yticks(np.arange(0.5, 1.51, step=0.5))
    plt.xticks(np.arange(9, 12.1, step=1))
    plt.tight_layout()
    if IMF == "Kroupa":
        plt.savefig('best_STF_Kroupa.pdf', dpi=250)
    if IMF == "igimf":
        plt.savefig('best_STF_igimf.pdf', dpi=250)


    fig = plt.figure(1002, figsize=(4, 3.5))
    plt.plot(*zip(*best_mass_SFEN_list_sorted), c='darkorange', ls='dashed')
    plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
    plt.ylabel(r'$t_{sf}$ [Gyr]')
    plt.xlim(9, 13)
    plt.ylim(0, 4)
    plt.plot(t, y1, c='b', ls='dotted', label='Thomas05')
    plt.plot(t, y3, c='g', ls='dashed', label='deLaRosa11')
    plt.plot(t4, y4, c='r', label='McDermid15')
    mass_thomas_list = np.arange(9, 12.1, 0.1)
    SFT_thomas_list = []
    for i in mass_thomas_list:
        # alpha_interpolate_obs = fun_obs_alpha_mass(i)
        alpha_interpolate_obs = -0.459 + 0.062 * i
        SFT_thomas = math.exp((1 / 5 - alpha_interpolate_obs) * 6)
        SFT_thomas_list.append(SFT_thomas)
    plt.plot(mass_thomas_list, SFT_thomas_list, c='k', ls='dotted', label='Thomas05-corrected')
    # plt.plot([], [], label='[Z/X], [Mg/Fe]:')
    plt.legend(prop={'size': 7}, loc='upper right')
    plt.tight_layout()
    # if IMF == "Kroupa":
    #     plt.savefig('best_SFEN_Kroupa.pdf', dpi=250)
    # if IMF == "igimf":
    #     plt.savefig('best_SFEN_igimf.pdf', dpi=250)


    fig = plt.figure(1003, figsize=(4, 3.5))
    plt.plot(*zip(*best_alpha_SFEN_list_sorted), c='darkorange', ls='dashed')
    plt.xlabel(r'$t_{sf}$ [Gyr]')
    plt.ylabel('[Mg/Fe]')
    plt.xlim(0, 4)
    plt.ylim(0, 0.65)
    SFT_thomas = np.arange(0.01, 4.1, 0.1)
    alpha_thomas = []
    for i in SFT_thomas:
        alpha_thomas.append(1 / 5 - 1 / 6 * math.log(i))
    plt.plot(SFT_thomas, alpha_thomas, c='k', ls='dashed', label='Thomas05')
    plt.legend(prop={'size': 7}, loc='best')
    plt.tight_layout()
    # if IMF == "Kroupa":
    #     plt.savefig('alpha_SFEN_Kroupa.pdf', dpi=250)
    # if IMF == "igimf":
    #     plt.savefig('alpha_SFEN_igimf.pdf', dpi=250)

    return









############# plot 3*3 array #############
#
#
#
#
# ########################
# ##### Fig A1 & A2 ###### # Run time: 572.9
# ########################
#
# def calculate_for_a_galaxy_mass(Dynamical_mass_k):
#     total_likelihood__ = 0
#     for STF_j__, SFEN_i__ in itertools.product(range(number_of_STF), range(number_of_SFEN)):
#         total_likelihood = calculate_likelihood_metal(SFEN_i__, STF_j__, Dynamical_mass_k, IMF, error_metal)
#         # total_likelihood = calculate_likelihood(SFEN_i__, STF_j__, Dynamical_mass_k, IMF, error_metal, error_alpha)[3]
#         if not np.isnan(total_likelihood):
#             if total_likelihood > total_likelihood__:
#                 total_likelihood__ = total_likelihood
#                 STF_j = STF_j__
#     SFEN_test_1 = round(number_of_SFEN / 10 * 8)
#     SFEN_test_2 = round(number_of_SFEN / 10 * 2)
#     total_likelihood_test_1 = calculate_likelihood(SFEN_test_1, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)[
#         3]
#     total_likelihood_test_2 = calculate_likelihood(SFEN_test_2, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)[
#         3]
#     # total_likelihood_test_1 = True
#     # total_likelihood_test_2 = True
#     if not (np.isnan(total_likelihood_test_1) and np.isnan(total_likelihood_test_2)):
#         total_likelihood__ = 0
#         SFEN__ = 0
#         for SFEN_i in range(number_of_SFEN):
#             SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood = \
#                 calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
#             SFEN_interpolate_list.append(SFEN_interpolate / 100)
#             STF_interpolate_list.append(STF_interpolate)
#             Dynamical_mass_interpolate_list.append(Dynamical_mass_interpolate)
#             total_likelihood_list.append(total_likelihood * prior_likelihood)
#             if not np.isnan(total_likelihood):
#                 if total_likelihood > total_likelihood__:
#                     total_likelihood__ = total_likelihood
#                     SFEN__ = SFEN_interpolate
#         if SFEN__ == 0:
#             best_mass_SFEN_list.append([None, None])
#         else:
#             best_mass_SFEN_list.append([Dynamical_mass_interpolate, SFEN__ / 100])
#     result = [SFEN_interpolate_list, STF_interpolate_list, Dynamical_mass_interpolate_list, total_likelihood_list,
#               best_mass_SFEN_list]
#     return result
#
#
# fig = plt.figure(10, figsize=(8, 6))
# ax = fig.add_subplot(111)
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
# ax.set_xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
# ax.set_ylabel(r'$t_{sf}$ [Gyr]')
# # clb = plt.colorbar(sc, shrink=1, aspect=40)
# # clb.set_label('likelihood')
# plot_number = 0
# for error_metal, error_alpha in itertools.product(error_metal_list, error_alpha_list):
#     print(error_metal, error_alpha)
#     prior_likelihood = 1  # math.erfc(abs(error_metal) / standard_deviation_systematic_metal / 2 ** 0.5) * \
#                        # math.erfc(abs(error_alpha) / standard_deviation_systematic_alpha / 2 ** 0.5)
#     plot_number = plot_number + 1
#     ax1 = fig.add_subplot(3, 3, plot_number)
#     SFEN_interpolate_list = []
#     STF_interpolate_list = []
#     Dynamical_mass_interpolate_list = []
#     total_likelihood_list = []
#     best_mass_SFEN_list = []  # [(mass1, SFEN1), (mass2, SFEN2)...]
#
#     pool = mp.Pool(mp.cpu_count())
#     results = pool.map(calculate_for_a_galaxy_mass,
#                        [Dynamical_mass_k for Dynamical_mass_k in range(number_of_Dynamical_mass)])
#     for i in range(number_of_Dynamical_mass):
#         # print(results[i])
#         SFEN_interpolate_list.extend(results[i][0])
#         STF_interpolate_list.extend(results[i][1])
#         Dynamical_mass_interpolate_list.extend(results[i][2])
#         total_likelihood_list.extend(results[i][3])
#         best_mass_SFEN_list.extend(results[i][4])
#     pool.close()
#     best_mass_SFEN_list_sorted = sorted(best_mass_SFEN_list, key=lambda l: l[0])
#
#     # print("Plotting...")
#     ax1.scatter(Dynamical_mass_interpolate_list, SFEN_interpolate_list, c=total_likelihood_list, s=10)
#     ax1.plot(*zip(*best_mass_SFEN_list_sorted), c='r', linewidth=0.8)
#     ax1.plot(t, y1, c='k', ls='dashed')
#     ax1.plot(t, y2, c='k', ls='-.')
#     plt.plot(t, y3, c='k', ls='dotted', label='deLaRosa11')
#     ax1.set_xlim(8.4, 12.1)
#     ax1.set_ylim(0, 8.1)
#     # if plot_number == 3 or plot_number == 6 or plot_number == 9:
#     #     clb = plt.colorbar(sc)
#     #     clb.set_label('likelihood')
#     ax1.set_title('[Z/X]_lit+({}); [Mg/Fe]_lit+({})'.format(error_metal, error_alpha), fontsize=8)
# plt.tight_layout()
# if IMF == "Kroupa":
#     plt.savefig('grid_Kroupa.pdf', dpi=250)
# if IMF == "igimf":
#     plt.savefig('grid_igimf.pdf', dpi=250)
#
#
#
# ################################
# #### Fig 3, 4, 5; 7, 8, 9 ######
# ################################
#
# def last_plot():
#     fig = plt.figure(1001, figsize=(4, 3.5))
#     plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
#     plt.ylabel(r'$f_{\rm st}$')
#     plt.xlim(9, 12)
#     plt.ylim(0, 1.5)
#     for error_metal, error_alpha in itertools.product(error_metal_list, error_alpha_list):
#         print(error_metal, error_alpha)
#
#         SFEN_interpolate_list = []
#         STF_interpolate_list = []
#         total_likelihood_list = []
#         best_mass_SFEN_list = []
#         best_STF_mass_list = []
#         best_alpha_SFEN_list = []
#         pool = mp.Pool(mp.cpu_count())
#         results = pool.map(parallel_mass, [Dynamical_mass_k for Dynamical_mass_k in range(number_of_Dynamical_mass)])
#         for i in range(number_of_Dynamical_mass):
#             SFEN_interpolate_list.extend(results[i][0])
#             STF_interpolate_list.extend(results[i][1])
#             total_likelihood_list.extend(results[i][2])
#             best_STF_mass_list.extend(results[i][3])
#             best_mass_SFEN_list.extend(results[i][4])
#             best_alpha_SFEN_list.extend(results[i][5])
#         pool.close()
#         best_mass_SFEN_list_sorted = sorted(best_mass_SFEN_list, key=lambda l: l[0])
#         best_alpha_SFEN_list_sorted = sorted(best_alpha_SFEN_list, key=lambda l: l[0])
#         best_STF_mass_list_sorted = sorted(best_STF_mass_list, key=lambda l: l[0])
#         plt.plot(*zip(*best_STF_mass_list_sorted), label='[Mg/Fe]-{}'.format(error_alpha))
#     plt.legend(prop={'size': 7}, loc='best')
#     plt.tight_layout()
#     if IMF == "Kroupa":
#         plt.savefig('best_STF_Kroupa.pdf', dpi=250)
#     if IMF == "igimf":
#         plt.savefig('best_STF_igimf.pdf', dpi=250)
        #
        # prior_likelihood = 1  #math.erfc(abs(error_metal) / standard_deviation_systematic_metal / 2 ** 0.5) * \
        #                    # math.erfc(abs(error_alpha) / standard_deviation_systematic_alpha / 2 ** 0.5)
        # SFEN_interpolate_list = []
        # STF_interpolate_list = []
        # Dynamical_mass_interpolate_list = []
        # total_likelihood_list = []
        # best_SFEN_list = []
        # dynamical_mass_list = []
        # alpha_interpolate_obs_list = []
        #
        # for Dynamical_mass_k in range(number_of_Dynamical_mass):
        #     total_likelihood__ = 0
        #     for STF_j__, SFEN_i__ in itertools.product(range(number_of_STF), range(number_of_SFEN)):
        #         total_likelihood = calculate_likelihood_metal(SFEN_i__, STF_j__, Dynamical_mass_k, IMF, error_metal)
        #         # total_likelihood = calculate_likelihood(SFEN_i__, STF_j__, Dynamical_mass_k, IMF, error_metal, error_alpha)[3]
        #         if not np.isnan(total_likelihood):
        #             if total_likelihood > total_likelihood__:
        #                 total_likelihood__ = total_likelihood
        #                 STF_j = STF_j__
        #     total_likelihood__ = 0
        #     SFEN__ = 0
        #     for SFEN_i in range(number_of_SFEN):
        #         SFEN_interpolate, STF_interpolate, Dynamical_mass_interpolate, total_likelihood = \
        #             calculate_likelihood(SFEN_i, STF_j, Dynamical_mass_k, IMF, error_metal, error_alpha)
        #         SFEN_interpolate_list.append(SFEN_interpolate / 100)
        #         STF_interpolate_list.append(STF_interpolate)
        #         Dynamical_mass_interpolate_list.append(Dynamical_mass_interpolate)
        #         total_likelihood_list.append(total_likelihood * prior_likelihood)
        #         if not np.isnan(total_likelihood):
        #             if total_likelihood > total_likelihood__:
        #                 total_likelihood__ = total_likelihood
        #                 SFEN__ = SFEN_interpolate
        #     if SFEN__ == 0:
        #         best_SFEN_list.append(None)
        #     else:
        #         # alpha_interpolate_obs = fun_obs_alpha_mass(Dynamical_mass_interpolate) + error_alpha
        #         alpha_interpolate_obs = -0.459 + 0.062 * Dynamical_mass_interpolate + error_alpha
        #         alpha_interpolate_obs_list.append(alpha_interpolate_obs)
        #         best_SFEN_list.append(SFEN__ / 100)
        #     dynamical_mass_list.append(Dynamical_mass_interpolate)
        # fig = plt.figure(1001, figsize=(4, 3.5))
        # plt.plot(Dynamical_mass_interpolate_list, STF_interpolate_list, label='[Mg/Fe]-{}'.format(error_alpha), c='k')
        #
        # fig = plt.figure(1002, figsize=(4, 3.5))
        # plt.plot(dynamical_mass_list, best_SFEN_list, c='r')
        # fig = plt.figure(1003, figsize=(4, 3.5))
        # plt.plot(best_SFEN_list, alpha_interpolate_obs_list, c='r')

    # fig = plt.figure(1002, figsize=(4, 3.5))
    # plt.xlabel(r'log$_{10}$($M_{\rm dyn}$ [M$_\odot$])')
    # plt.ylabel(r'$t_{sf}$ [Gyr]')
    # plt.xlim(9, 13)
    # plt.ylim(0, 4)
    # plt.plot([], [], label='[Z/X] {}'.format(error_metal_list[0]), c='k', ls='dashed')
    # plt.plot([], [], label='[Z/X] {}'.format(error_metal_list[1]), c='k')
    # plt.plot([], [], label='[Z/X] {}'.format(error_metal_list[2]), c='k', ls='dotted')
    # plt.plot([], [], label='[Mg/Fe] {}'.format(error_alpha_list[0]), c='r')
    # plt.plot([], [], label='[Mg/Fe] {}'.format(error_alpha_list[1]), c='g')
    # plt.plot([], [], label='[Mg/Fe] {}'.format(error_alpha_list[2]), c='b')
    # plt.plot(t, y1, c='k', ls='dashed', label='Thomas05-ln')
    # plt.plot(t, y2, c='k', ls='-.', label='Recchi09')
    # plt.plot(t, y3, c='k', ls='dotted', label='deLaRosa11')
    # mass_thomas_list = np.arange(9, 12.1, 0.1)
    # SFT_thomas_list = []
    # for i in mass_thomas_list:
    #     # alpha_interpolate_obs = fun_obs_alpha_mass(i)
    #     alpha_interpolate_obs = -0.459+0.062*i
    #     SFT_thomas = math.exp((1/5-alpha_interpolate_obs)*6)
    #     SFT_thomas_list.append(SFT_thomas)
    # plt.plot(mass_thomas_list, SFT_thomas_list, c='k', ls='dotted', label='Thomas05-ln')
    # # plt.plot([], [], label='[Z/X], [Mg/Fe]:')
    # plt.legend(prop={'size': 7}, loc='upper right')
    # plt.tight_layout()
    # if IMF == "Kroupa":
    #     plt.savefig('best_SFEN_Kroupa.pdf', dpi=250)
    # if IMF == "igimf":
    #     plt.savefig('best_SFEN_igimf.pdf', dpi=250)
    #
    #
    # fig = plt.figure(1003, figsize=(4, 3.5))
    # plt.xlabel(r'$t_{sf}$ [Gyr]')
    # plt.ylabel('[Mg/Fe]')
    # plt.xlim(0, 4)
    # plt.ylim(0, 0.65)
    # plt.plot([], [], label='[Z/X] {}'.format(error_metal_list[0]), c='k', ls='dashed')
    # plt.plot([], [], label='[Z/X] {}'.format(error_metal_list[1]), c='k')
    # plt.plot([], [], label='[Z/X] {}'.format(error_metal_list[2]), c='k', ls='dotted')
    # plt.plot([], [], label='[Mg/Fe] {}'.format(error_alpha_list[0]), c='r')
    # plt.plot([], [], label='[Mg/Fe] {}'.format(error_alpha_list[1]), c='g')
    # plt.plot([], [], label='[Mg/Fe] {}'.format(error_alpha_list[2]), c='b')
    # SFT_thomas = np.arange(0.01, 4.1, 0.1)
    # alpha_thomas = []
    # for i in SFT_thomas:
    #     alpha_thomas.append(1/5-1/6*math.log(i))
    # plt.plot(SFT_thomas, alpha_thomas, c='k', ls='dashed', label='Thomas05')
    # plt.legend(prop={'size': 7}, loc='best')
    # plt.tight_layout()
    # if IMF == "Kroupa":
    #     plt.savefig('alpha_SFEN_Kroupa.pdf', dpi=250)
    # if IMF == "igimf":
    #     plt.savefig('alpha_SFEN_igimf.pdf', dpi=250)
    return

# def plot_dirction():
#
#     plt.plot(*zip(*plot_ZX_SFT), c='b', zorder=8)
#     plt.plot(*zip(*plot_ZX_STF), c='g', zorder=8)
#     plt.plot(*zip(*plot_ZX_SFR), c='r', zorder=8)
#     plt.plot(*zip(*plot_MgFe_SFT), ls='dashed', c='b', zorder=8)
#     plt.plot(*zip(*plot_MgFe_STF), ls='dashed', c='g', zorder=8)
#     plt.plot(*zip(*plot_MgFe_SFR), ls='dashed', c='r', zorder=8)
#
#     plt.scatter(*zip(*plot_ZX_SFT), c='b', zorder=9, s=20, label=r'$t_{\rm sf}$')
#     plt.scatter(*zip(*plot_ZX_STF), c='g', zorder=9, s=20, label=r'$f_{\rm st}$')
#     plt.scatter(*zip(*plot_ZX_SFR), c='r', zorder=9, s=20, label=r'log$_{10}(SFR)$')
#     plt.scatter(*zip(*plot_MgFe_SFT), c='b', zorder=9, s=20)
#     plt.scatter(*zip(*plot_MgFe_STF), c='g', zorder=9, s=20)
#     plt.scatter(*zip(*plot_MgFe_SFR), c='r', zorder=9, s=20)
#
#     plt.scatter([plot_ZX_SFR[2][0]], [plot_ZX_SFR[2][1]], c='k', zorder=9, s=20)
#     plt.scatter([plot_MgFe_SFR[2][0]], [plot_MgFe_SFR[2][1]], c='k', zorder=9, s=20)
#
#     # plt.plot(*zip(*plot_ZX_SFT_igimf))
#     # plt.plot(*zip(*plot_ZX_STF_igimf))
#     # plt.plot(*zip(*plot_ZX_SFR_igimf))
#     # plt.plot(*zip(*plot_MgFe_SFT_igimf))
#     # plt.plot(*zip(*plot_MgFe_STF_igimf))
#     # plt.plot(*zip(*plot_MgFe_SFR_igimf))
#
#     for a_point in plot_ZX_SFT_label:
#         plt.annotate(round(a_point[2])/100, (a_point[0]+0.03, a_point[1]+0.03), fontsize=9, color='b', zorder=10)
#     for a_point in plot_ZX_STF_label:
#         plt.annotate(a_point[2], (a_point[0]+0.04, a_point[1]-0.07), fontsize=9, color='g', zorder=10)
#     for a_point in plot_ZX_SFR_label:
#         plt.annotate(round(a_point[2]), (a_point[0]-0.13, a_point[1]+0.02), fontsize=9, color='r', zorder=10)
#     for a_point in plot_MgFe_SFT_label:
#         plt.annotate(round(a_point[2])/100, (a_point[0]-0.15, a_point[1]-0.1), fontsize=9, color='b', zorder=10)
#     for a_point in plot_MgFe_STF_label:
#         plt.annotate(a_point[2], (a_point[0]+0.1, a_point[1]+0.01), fontsize=9, color='g', zorder=10)
#     for a_point in plot_MgFe_SFR_label:
#         plt.annotate(round(a_point[2]), (a_point[0]-0.13, a_point[1]+0.015), fontsize=9, color='r', zorder=10)
#
#     plt.xlim(9, 12)
#     plt.legend(prop={'size': 8})
#     plt.tight_layout()
#     plt.savefig('dirction.pdf', dpi=250)
#     return





if __name__ == '__main__':
    total_start = time()

    # ######### assume typical systematic error in observation to be 0.1 dex (see Conroy et al. 2014) #########
    # standard_deviation_systematic_metal = 0.1
    # standard_deviation_systematic_alpha = 0.1
    ######### and setup modifications on the observational abundances #########
    error_metal = 0
    # error_alpha = 0.23
    error_alpha = 0
    # error_alpha = -0.15
    IMF = "Kroupa"
    # IMF = "igimf"
    error_metal_list = [0]
    # error_alpha_list = [0.23]
    # error_alpha_list = [-0.15]
    error_alpha_list = [0]



    SFEN_select = True
    STF_select = True
    # SFEN_select = 100
    # STF_select = 0.6
    (points, points_igimf, metal_values, metal_values_igimf, alpha_values, alpha_values_igimf, Dynamical_mass_igimf,
     Mass_weighted_stellar_Z_over_X_igimf, Mass_weighted_stellar_Mg_over_Fe_igimf, Dynamical_mass,
     Mass_weighted_stellar_Z_over_X, Mass_weighted_stellar_Mg_over_Fe,
     plot_ZX_SFT, plot_ZX_STF, plot_ZX_SFR, plot_MgFe_SFT, plot_MgFe_STF, plot_MgFe_SFR,
     plot_ZX_SFT_label, plot_ZX_STF_label, plot_ZX_SFR_label, plot_MgFe_SFT_label, plot_MgFe_STF_label,
     plot_MgFe_SFR_label, STF, STF_igimf, SFEN, SFEN_igimf, Mass_weighted_stellar_Fe_over_H_igimf,
     plot_ZX_SFT_igimf, plot_MgFe_SFT_igimf, plot_ZX_STF_igimf, plot_MgFe_STF_igimf, plot_ZX_SFR_igimf,
     plot_MgFe_SFR_igimf, plot_ZX_SFT_label_igimf, plot_ZX_STF_label_igimf, plot_ZX_SFR_label_igimf,
     plot_MgFe_SFT_label_igimf, plot_MgFe_STF_label_igimf, plot_MgFe_SFR_label_igimf,
     AAA_igimf, error_if_AAA_is_1, Log_SFR_igimf) = impot_simulation(SFEN_select)

    # grid_SFEN, grid_STF, grid_Dynamical_mass = np.mgrid[5:400:99j, 0.1:1.5:99j, 9:12:99j]
    # grid_STF_list = np.mgrid[0.1:1.5:99j]
    grid_SFEN, grid_STF, grid_Dynamical_mass = np.mgrid[1:400:99j, 0.1:1:99j, 8:12:99j]
    grid_STF_list = np.mgrid[0.1:1.0:99j]
    # the range of grid_Dynamical_mass is limited by the observation data
    number_of_SFEN = 99
    number_of_STF = 99
    number_of_Dynamical_mass = 99

    (grid_alpha1, grid_metal1, grid_alpha_igimf1, grid_metal_igimf1) = interpolate_simulation(points, points_igimf,
         metal_values, metal_values_igimf, alpha_values, alpha_values_igimf, grid_SFEN, grid_STF, grid_Dynamical_mass)



    ##################################################################

    # plot_obs(plot_data=True, plot_dirction=False)

    ##### plot_dirction()

    plt.xlim(9, 120)
    Mass_weighted_stellar_Z_over_X_error = []
    Mass_weighted_stellar_Z_over_X_igimf_error = []
    Mass_weighted_stellar_Mg_over_Fe_error = []
    Mass_weighted_stellar_Mg_over_Fe_igimf_error = []
    for item in Mass_weighted_stellar_Z_over_X:
        Mass_weighted_stellar_Z_over_X_error.append(item-error_metal)
    for item in Mass_weighted_stellar_Z_over_X_igimf:
        Mass_weighted_stellar_Z_over_X_igimf_error.append(item-error_metal)
    for item in Mass_weighted_stellar_Mg_over_Fe:
        Mass_weighted_stellar_Mg_over_Fe_error.append(item-error_alpha)
    for item in Mass_weighted_stellar_Mg_over_Fe_igimf:
        Mass_weighted_stellar_Mg_over_Fe_igimf_error.append(item-error_alpha)
    plot_simulation(IMF, Dynamical_mass, Dynamical_mass_igimf, Mass_weighted_stellar_Z_over_X_error,
                    Mass_weighted_stellar_Z_over_X_igimf_error, Mass_weighted_stellar_Mg_over_Fe_error,
                    Mass_weighted_stellar_Mg_over_Fe_igimf_error, STF, STF_igimf, SFEN, SFEN_igimf,
                    Mass_weighted_stellar_Fe_over_H_igimf, plot_ZX_SFT_igimf, plot_MgFe_SFT_igimf, plot_ZX_STF_igimf,
                    plot_MgFe_STF_igimf, plot_ZX_SFR_igimf, plot_MgFe_SFR_igimf, plot_ZX_SFT_label_igimf,
                    plot_ZX_STF_label_igimf, plot_ZX_SFR_label_igimf, plot_MgFe_SFT_label_igimf,
                    plot_MgFe_STF_label_igimf, plot_MgFe_SFR_label_igimf, AAA_igimf, error_if_AAA_is_1, Log_SFR_igimf)

    # plot_likelihood()  # reproduce Thomas05 with fixed STF  # change line 550 for fitting only with [Mg/Fe] observation constraints
    #
    # plot_Update()

    # plot_fig_likelihood()  # free STF

    # plot_best_fits()

    # last_plot()  # not used?

    # plot_age_limit() # not used.

    # plt.tight_layout()

    # middle_start = time()
    # middle_end = time()
    # print("middle time pool:", middle_end - middle_start)
    total_end = time()
    print("Run time:", total_end - total_start)
    # plt.savefig('best_fit.pdf', dpi=250)
    plt.show()