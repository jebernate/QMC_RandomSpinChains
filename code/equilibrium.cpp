#include "include/SSE_Heisenberg1D.hpp"
#include <omp.h>

int main(int argc, char *argv[])
{
	std::cout << "-----------------------------------------------------------------------------------\n";
	std::cout << "| Stochastic Series Expansion Simulation of Random Heisenberg Chains: Equilibrium |\n";
	std::cout << "-----------------------------------------------------------------------------------\n";

	int L = 200;
	int noise_samples = 500;
	int n_max = 6;
	double beta = std::pow(2, n_max);
	double W = 0.75;
	int MCS_run_eq = 300; // Monte Carlo sweeps

	long int init_seed = 999;
	long int sampling_seed = 123;

	std::cout << "------------------------ SIMULATION PARAMETERS -------------------------\n";

	std::cout << std::setw(50) << std::left << "* Chain length:" << std::setw(25) << std::left << L << "\n";
	std::cout << std::setw(50) << std::left << "* Box distribution of bonds (W):" << std::setw(25) << std::left << W << "\n";
	std::cout << std::setw(50) << std::left << "* Equilibration:" << std::setw(25) << std::left << "Single-run" << "\n";
	std::cout << std::setw(50) << std::left << "* Maximum Monte Carlo Sweeps for equilibrium:" << std::setw(25) << std::left << MCS_run_eq << "\n";
	std::cout << std::setw(50) << std::left << "* Maximum beta to be reached: " << std::setw(25) << std::left << beta << "\n";
	std::cout << std::setw(50) << std::left << "* Noise samples:" << std::setw(25) << std::left << noise_samples << "\n";

	std::cout << "------------------------------------------------------------------------\n";

	auto start = std::chrono::steady_clock::now();
	int counter = 1;

	// Threads

	int NUM_THREADS = 6;
	if (NUM_THREADS > omp_get_max_threads())
		NUM_THREADS = omp_get_max_threads();
	std::cout << "Sampling with " << NUM_THREADS << " / " << omp_get_max_threads() << " threads.\n\n";

	// Simulation

	double sumE = 0.0;
	std::vector<double> sum_run_nh(MCS_run_eq, 0.0);
	std::vector<double> sum_run_zz_corr_mid(MCS_run_eq, 0.0);

#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < noise_samples; i++)
	{
		Heisenberg1D ham(W, L, init_seed * (i + 2023));
		ham.equilibrium(beta, MCS_run_eq, sampling_seed * (i + 2024));
#pragma omp critical
		{
			// Update accumulators of correlation for all l
			plus_equal_vector(sum_run_zz_corr_mid, ham.run_zz_corr_mid);
			// Update accumulators of correlation at L/2 for various beta
			plus_equal_vector(sum_run_nh, ham.run_nh);
		}
#pragma omp critical
		if (counter % 10 == 0)
		{
			std::cout << "Sampling...\t" << counter << " / " << noise_samples << " \t";
			elapsed(counter, start);
		}
#pragma omp atomic
		counter++;
	}

	scale_vector(sum_run_zz_corr_mid, 1.0 / noise_samples);
	scale_vector(sum_run_nh, 1.0 / noise_samples);

	std::string folder = "results/";
	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << W;
	std::string W_str = stream.str();

	std::string out_filename = folder + "eq_L_" + std::to_string(L) + "_W_" + W_str + "_beta_" + std::to_string((int)beta);
	out_filename += "_MCS_" + std::to_string(MCS_run_eq) + "_noise_sam_" + std::to_string(noise_samples) + ".txt";

	std::cout << "Saving to " << out_filename << "\n";

	saveVector(sum_run_zz_corr_mid, out_filename, false);
	saveVector(sum_run_nh, out_filename, true);

	return 0;
}