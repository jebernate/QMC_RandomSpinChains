"""Code to analyse the results of the quantum Monte Carlo simulation
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})


def process_data_corr(corr: np.ndarray, delta_corr: np.ndarray, cutoff: int):
    """Separates the correlation data on l even and odd

    Args:
        corr (np.ndarray): raw correlation data
        delta_corr (np.ndarray): error bards of the previous
        cutoff (int): Values above the cutoff are not plotted

    """

    l = np.arange(1, corr.shape[0] + 1)
    cond = l < cutoff
    # delta_corr = delta_corr[cond]
    corr_cond = corr[cond]
    corr_odd = corr_cond[::2]
    # delta_corr_odd = delta_corr[::2]
    corr_even = corr_cond[1::2]
    # delta_corr_even = delta_corr[1::2]
    l = l[cond]
    l_even = l[1::2]
    l_odd = l[::2]

    return l, l_even, l_odd, corr_even, corr_odd


# STRUCTURE FACTOR


def calculate_structure_factor(correlation: np.ndarray) -> np.ndarray:
    """Calculates the structure factor as the fourier transform of the correlation

    Args:
        correlation (np.ndarray): zz correlation function from 0 to L/2

    Returns:
        np.ndarray: Array containing the structure factor in the interval q [-pi, pi]
    """

    length = correlation.shape[0]
    correlation = np.append(correlation, correlation[:-1][::-1])
    correlation = np.insert(correlation, 0, 0.25)

    structure_factor = np.zeros(length, dtype=complex)
    q = np.linspace(0, np.pi, length)
    for k in range(length):
        structure_factor[k] = (
            2 * np.pi * np.sum(correlation * np.exp(-1j * q[k] * np.arange(2 * length)))
        )
    return q, structure_factor


def plot_structure_factor():
    """Calculate and plot the structure factor given the correlation data"""
    corr_200, delta_corr_200 = np.loadtxt(
        "results/Corr_L_200_W_0.75_nmax_16_MCS_25.txt", delimiter=","
    )
    corr_100, delta_corr_100 = np.loadtxt(
        "results/Corr_L_100_W_1.00_nmax_17_MCS_20.txt", delimiter=","
    )

    q_200, ft_200 = calculate_structure_factor(corr_200)
    q_100, ft_100 = calculate_structure_factor(corr_100)
    fig, ax = plt.subplots()
    ax.plot(
        q_200 / np.pi,
        np.real(ft_200) / np.pi,
        "b:o",
        markersize=3,
        label=r"$W = 0.75\,(L=200)$",
    )
    ax.plot(
        q_100 / np.pi,
        np.real(ft_100) / np.pi,
        "r:o",
        markersize=3,
        label=r"$W = 1.0\,(L=100)$",
    )
    ax.plot(
        q_200 / np.pi,
        np.pi**2 / 12 * q_200 / np.pi,
        "k--",
        label=r"SDRG, $\kappa|q|$",
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 2.5])
    ax.set_xlabel(r"$q / \pi$")
    ax.set_ylabel(r"$S^z(q) / \pi$")
    ax.legend()
    plt.savefig("images/structure_factor.jpg", bbox_inches="tight", dpi=210)


# EQUILIBRIUM


def plot_eq():
    """Plot the nh value (number of non-identiy hamiltonian terms in the configuration being sampled), together with
    the correlation function at L/2 for values equilibrium steps between 0 and 300 MCS.
    """
    corr_mid, nh = np.loadtxt(
        "results/eq_L_200_W_0.75_beta_65536_MCS_300_noise_sam_500.txt", delimiter=","
    )

    nh_mean, nh_std = np.mean(nh[-50:]), np.std(nh[-50:])
    corr_mid_mean, corr_mid_std = np.mean(corr_mid[-50:]), np.std(corr_mid[-50:])

    fig, ax = plt.subplots()
    ax.plot(nh)
    ax.axhline(nh_mean, linestyle="--", color="k", label=r"$\langle n_h\rangle_{eq}$")
    ax.axhline(
        nh_mean + 3 * nh_std,
        linestyle="--",
        color="r",
        label=r"$\langle n_h\rangle_{eq} \pm 3 \sigma$",
    )
    ax.axhline(nh_mean - 3 * nh_std, linestyle="--", color="r")
    ax.set_yscale("log")
    ax.set_ylabel(r"$n_h$")
    ax.set_xlabel("Monte Carlo Sweeps")
    ax.grid()
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(corr_mid)
    ax.axhline(
        corr_mid_mean,
        linestyle="--",
        color="k",
        label=r"$\langle C^{zz}(L/2)\rangle_{eq}$",
    )
    ax.axhline(
        corr_mid_mean + 3 * corr_mid_std,
        linestyle="--",
        color="r",
        label=r"$\langle C^{zz}(L/2)\rangle_{eq} \pm 3\sigma$",
    )
    ax.axhline(corr_mid_mean - 3 * corr_mid_std, linestyle="--", color="r")
    ax.set_ylabel(r"$C^{zz}(L/2)$")
    ax.set_xlabel("Monte Carlo Sweeps")
    ax.grid()
    ax.legend()
    plt.show()


# CORRELATIONS


def plot_correlations():
    """Load correlation data and plot the correlation function at even and odd l, and the sum of even and odd"""
    # Load the data

    corr_200, delta_corr_200 = np.loadtxt(
        "results/Corr_L_200_W_0.75_nmax_16_MCS_25.txt", delimiter=","
    )
    corr_100, delta_corr_100 = np.loadtxt(
        "results/Corr_L_100_W_1.00_nmax_17_MCS_20.txt", delimiter=","
    )

    # Process the data

    l_200, l_even_200, l_odd_200, corr_even_200, corr_odd_200 = process_data_corr(
        corr_200, delta_corr_200, 60
    )
    l_100, l_even_100, l_odd_100, corr_even_100, corr_odd_100 = process_data_corr(
        corr_100, delta_corr_100, 30
    )

    # Correlation as a function of l

    fig, ax = plt.subplots()
    ax.plot(l_even_200, corr_even_200, "k:s", markersize=4, label=r"$l$ even")
    ax.plot(l_odd_200, np.abs(corr_odd_200), "r:o", markersize=4, label=r"$l$ odd")

    # For comparison purposes
    ax.plot(l_200, 0.9 / l_200**2, "--k", label=r"$0.9/l^2$ (even)")
    ax.plot(l_200, 0.98 / l_200**2, "--r", label=r"$0.98/l^2$ (odd)")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("$l$")
    ax.set_ylabel(r"$|C^{zz}(l)|$")
    plt.savefig("images/corr_even_odd_075.jpg", bbox_inches="tight", dpi=220)
    # plt.show()

    ## Sum of even and odd correlations

    corr_sum_200 = corr_even_200 + corr_odd_200[:-1]
    # delta_corr_sum_200 = delta_corr_even_200 + delta_corr_odd_200[:-1]
    l_sum_200 = np.arange(1, l_even_200[-1] + 1, 2)
    fig, ax = plt.subplots()
    ax.plot(
        l_sum_200,
        np.abs(corr_sum_200),
        ":s",
        markersize=4,
        label=r"$W = 0.75\, (L = 200)$",
    )

    corr_sum_100 = corr_even_100 + corr_odd_100[:-1]
    # delta_corr_sum_100 = delta_corr_even_100 + delta_corr_odd_100[:-1]
    l_sum_100 = np.arange(1, l_even_100[-1] + 1, 2)
    ax.plot(
        l_sum_100,
        np.abs(corr_sum_100),
        ":s",
        markersize=4,
        label=r"$W = 1.0\,(L = 100)$",
    )

    # For comparison with SDRG prediction
    ax.plot(
        l_sum_200[l_sum_200 > 10],
        1 / 12 / l_sum_200[l_sum_200 > 10] ** 2,
        "k--",
        label=r"$1/12 l^{-2}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$l$")
    ax.set_ylabel(r"$|C^{zz}(l) + C^{zz}(l+1)|$")
    ax.legend()
    plt.savefig("images/corr_sum_100_200.jpg", bbox_inches="tight", dpi=220)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)

    plot_eq()
    # plot_correlations()
    # plot_structure_factor()
