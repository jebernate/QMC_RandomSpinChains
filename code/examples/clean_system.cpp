#include "../include/SSE_Heisenberg1D.hpp"
#include <string.h>

std::vector<double> jackknife_analysis_energy(const std::vector<double> &data_vector, int M_bins, double beta, int L)
{
	int samples_per_bin = data_vector.size() / M_bins;
	double E = 0.0, dE = 0.0;

	std::vector<double> E_bins(M_bins, 0.0);
	std::vector<double> U(M_bins, 0.0);

	int j_prev = 0;
	for (int i = 0; i < M_bins; i++)
	{
		for (int j = 0; j < samples_per_bin; j++)
		{
			E_bins[i] += data_vector[j_prev + j];
		}
		j_prev += samples_per_bin;
	}
	for (int i = 0; i < M_bins; i++)
	{
		for (int j = 0; j < M_bins; j++)
		{
			if (j != i)
			{
				U[i] += E_bins[j];
			}
		}
		U[i] = 0.25 - U[i] / (samples_per_bin * (M_bins - 1) * beta * L);
		E += U[i];
	}
	E /= M_bins;

	double sum_sq_res = 0.0; // Sum of the squared residues
	for (int i = 0; i < M_bins; i++)
	{
		sum_sq_res += (U[i] - E) * (U[i] - E);
	}

	// Final estimator

	dE = std::sqrt((M_bins - 1.0) / M_bins * sum_sq_res);

	return std::vector<double>{E, dE};
}

int main(int argc, char *argv[])
{
	std::cout << "----------------------------------------------------------------------\n";
	std::cout << "| Stochastic Series Expansion Simulation of Random Heisenberg Chains |\n";
	std::cout << "----------------------------------------------------------------------\n";

	int L = 20;
	int MCS_eq = 200;
	int MCS_sa = 100;
	int n_max = 16;
	double beta = std::pow(2, n_max);
	long int sampling_seed = 123456;

	std::cout << "-------------------- SIMULATION PARAMETERS ---------------------\n";

	std::cout << std::setw(45) << std::left << "* Chain length:" << std::setw(25) << std::left << L << "\n";
	std::cout << std::setw(45) << std::left << "* Box distribution of bonds (W):" << std::setw(25) << std::left << 0.0 << "\n";
	std::cout << std::setw(45) << std::left << "* Equilibration mode: " << std::setw(25) << std::left << "Uniform" << "\n";
	std::cout << std::setw(45) << std::left << "* Monte Carlo Steps for equilirbium:" << std::setw(25) << std::left << MCS_eq << "\n";
	std::cout << std::setw(45) << std::left << "* Monte Carlo Steps for sampling:" << std::setw(25) << std::left << MCS_sa << "\n";
	std::cout << std::setw(45) << std::left << "* Beta: " << std::setw(25) << std::left << beta << "\n";

	std::cout << "----------------------------------------------------------------\n";

	// Simulation

	Heisenberg1D ham(0.0, L, 0);

	auto start = std::chrono::steady_clock::now();
	ham.run(beta, MCS_eq, MCS_sa, sampling_seed);

	// Analyse energy data using a simple binning procedure or Jackknife

	int M_bins = 10;
	std::vector<double> E = jackknife_analysis_energy(ham.run_nh, M_bins, beta, L);
	std::cout << "Ground state energy: " << E[0] << " +/- " << E[1] << "\n";

	auto time = std::chrono::steady_clock::now() - start;
	std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::seconds>(time).count() << " seconds\n";

	std::string folder = "results/";
	std::string out_filename = folder + "Corr_L_" + std::to_string(L) + "_clean_nmax_" + std::to_string(n_max);
	out_filename += "_MCS_eq_" + std::to_string(MCS_eq) + "_MCS_sa_" + std::to_string(MCS_sa) + ".txt";

	std::cout << "Saving to: " << out_filename << "\n";

	saveVector(ham.sample_zz_corr, out_filename, false);
	saveVector(ham.run_nh, out_filename, true);

	return 0;
}