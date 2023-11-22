#include "include/SSE_Heisenberg1D.hpp"
#include <omp.h>
#include <string.h>
#include <sstream>

int main(int argc, char *argv[])
{
	std::cout << "----------------------------------------------------------------------\n";
	std::cout << "| Stochastic Series Expansion Simulation of Random Heisenberg Chains |\n";
	std::cout << "----------------------------------------------------------------------\n";

	std::string filename = argv[1];
	std::ifstream inputFile(filename);
	if (!inputFile.is_open())
	{
		std::cerr << "Failed to open the file for reading: " << filename << std::endl;
	}

	std::vector<std::string> data;
	std::string line;
	while (std::getline(inputFile, line))
	{
		data.push_back(line);
	}
	inputFile.close();

	int L = std::stoi(data.at(0));
	double W = std::stod(data.at(1));
	int noise_samples = std::stoi(data.at(2));
	int n_max = std::stoi(data.at(3));
	int MCS_beta = std::stoi(data.at(4));

	double beta = std::pow(2, n_max);
	long int init_seed = 987654;
	long int sampling_seed = 12345;

	std::vector<double> sum_zz_corr(L / 2, 0.0);
	std::vector<double> sample_zz_corr2(L / 2, 0.0);
	std::vector<double> delta_sum_zz_corr(L / 2, 0.0);

	std::vector<double> sum_zz_corr_mid(n_max + 1, 0.0);
	std::vector<double> zz_corr_mid2(n_max + 1, 0.0);
	std::vector<double> delta_sum_zz_corr_mid(n_max + 1, 0.0);

	std::cout << "-------------------- SIMULATION PARAMETERS ---------------------\n";

	std::cout << std::setw(45) << std::left << "* Chain length:" << std::setw(25) << std::left << L << "\n";
	std::cout << std::setw(45) << std::left << "* Box distribution of bonds (W):" << std::setw(25) << std::left << W << "\n";
	std::cout << std::setw(45) << std::left << "* Equilibration:" << std::setw(25) << std::left << "Beta-doubling" << "\n";
	std::cout << std::setw(45) << std::left << "* Monte Carlo Steps per beta:" << std::setw(25) << std::left << MCS_beta << "\n";
	std::cout << std::setw(45) << std::left << "* Maximum beta to be reached: " << std::setw(25) << std::left << beta << "\n";
	std::cout << std::setw(45) << std::left << "* Noise samples:" << std::setw(25) << std::left << noise_samples << "\n";

	std::cout << "----------------------------------------------------------------\n";

	auto start = std::chrono::steady_clock::now();
	int counter = 1;

	// Set the number of threads

	int NUM_THREADS = 7;
	if (NUM_THREADS > omp_get_max_threads())
		NUM_THREADS = omp_get_max_threads();
	std::cout << "Sampling with " << NUM_THREADS << " / " << omp_get_max_threads() << " threads.\n\n";

	// Simulation

#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < noise_samples; i++)
	{
		Heisenberg1D ham(W, L, init_seed * (i + 9999));
		ham.beta_doubling(n_max, MCS_beta, sampling_seed * (i + 1111));
		// ham.run(beta, 150, 10, sampling_seed * (i + 1111)); // Also possible to use simple 'thermalization'
#pragma omp critical
		{
			// Update accumulators of correlation for all l
			plus_equal_vector(sum_zz_corr, ham.sample_zz_corr);
			sum_square(delta_sum_zz_corr, ham.sample_zz_corr);
			// Update accumulators of correlation at L/2 for various beta
			plus_equal_vector(sum_zz_corr_mid, ham.sample_zz_corr_mid);
			sum_square(delta_sum_zz_corr_mid, ham.sample_zz_corr_mid);
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

	// Divide by the total number of samples

	scale_vector(sum_zz_corr, 1.0 / noise_samples);
	scale_vector(sum_zz_corr_mid, 1.0 / noise_samples);

	// Compute the variance
	for (int i = 0; i < L / 2; i++)
	{
		delta_sum_zz_corr[i] = 1.0 / std::sqrt(noise_samples) * std::sqrt(delta_sum_zz_corr[i] / noise_samples - sum_zz_corr[i] * sum_zz_corr[i]);
	}
	for (int i = 0; i <= n_max; i++)
	{
		delta_sum_zz_corr_mid[i] = 1.0 / std::sqrt(noise_samples) * std::sqrt(delta_sum_zz_corr_mid[i] / noise_samples - sum_zz_corr_mid[i] * sum_zz_corr_mid[i]);
	}

	std::string folder = "results/";
	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << W;
	std::string W_str = stream.str();

	std::string out_filename = folder + "Corr_L_" + std::to_string(L) + "_W_" + W_str + "_nmax_" + std::to_string(n_max);
	out_filename += "_MCS_" + std::to_string(MCS_beta) + ".txt";

	std::cout << "Saving to: " << out_filename << "\n";

	saveVector(sum_zz_corr, out_filename, false);
	saveVector(delta_sum_zz_corr, out_filename, true); // Append to the previous file

	std::string out_filename_mid = folder + "CorrMid_L_" + std::to_string(L) + "_W_" + W_str + "_nmax_" + std::to_string(n_max);
	out_filename_mid += "_MCS_" + std::to_string(MCS_beta) + ".txt";

	saveVector(sum_zz_corr_mid, out_filename_mid, false);
	saveVector(delta_sum_zz_corr_mid, out_filename_mid, true); // Append to the previous file

	std::cout << "Saving to: " << out_filename_mid << "\n";

	return 0;
}
