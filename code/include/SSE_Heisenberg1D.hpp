#include "misc.hpp"
#include <cmath>
#include <numeric>
#include <random>

class Heisenberg1D
{
private:

	int n_bonds;
	std::vector<int> spin;
	std::vector<double> J;

	std::vector<int> first_op;
	std::vector<int> last_op;
	std::vector<int> op_string;
	std::vector<int> vertex_list;

	// Observables

	std::vector<double> zz_corr;

	// SSE algorithm functions

	void DiagonalUpdate(double beta, std::mt19937 &gen);
	void CreateVertexList(void);
	void LoopUpdate(double beta, std::mt19937 &gen);
	void AdjustCutoff(void);
	void DoubleSystem(void);
	void clear(void);
	void shrink(void);
	double ComputeCorrMid(void);

public:

	Heisenberg1D(double W, int chain_length, long int init_seed)
	{

		L = chain_length;
		n_bonds = chain_length;

		// Set up the maximum string length (to be adjusted)

		M = chain_length;
		nh = 0;

		// Inititalize the spin chain and couplings J

		std::uniform_real_distribution<> dist(0.0, 1.0);
		std::mt19937 gen(init_seed);

		J.assign(n_bonds + 1, 0.0);
		spin.assign(n_bonds + 1, 1);
		for (int i = 1; i <= n_bonds; i++)
		{
			J[i] = 1.0 + W * (2 * dist(gen) - 1.0);
			spin[i] = (int) std::pow(-1.0, i);
		}
		// Initialize all the operators to the identity i.e 0

		op_string.assign(M, 0);

		// Allocate memory for vertex_list, first_op and last_op

		vertex_list.assign(4 * M, -1);
		first_op.assign(L + 1, -1);
		last_op.assign(L + 1, -1);

		// Observables

		zz_corr.assign(L / 2, 0.0);
	}

	int L;
	int M, M_prev;
	int nh;
	double E, dE;
	
	// Observables

	std::vector<double> sample_zz_corr;
	std::vector<double> sample_zz_corr_mid;
	std::vector<double> run_zz_corr_mid;
	std::vector<double> run_nh;

	// Methods for simulating: include the standard equilibration routine and the beta-doubling scheme

	void run(double beta, int MCS_eq, int MCS_sa, long int seed_eq);
	void beta_doubling(int n_max, int MCS_eq, long int seed_eq);
	void equilibrium(double beta, int MCS_run_eq, long int seed_eq);
	void ComputeCorr(void);
};

void Heisenberg1D::DiagonalUpdate(double beta, std::mt19937 &gen)
{

	std::uniform_real_distribution<double> dist(0.0, 1.0);

	int b, op;
	int s1, s2;

	double prob_insert = 0.5 * beta * n_bonds;
	double prob_remove = 2.0 / (beta * n_bonds);

	for (int i = 0; i < M; i++)
	{
		op = op_string[i];
		// If there is an identity, add a diagonal operator
		if (op == 0)
		{
			// b = 1 + n_bonds * Ran64.r();
			b = 1 + n_bonds * dist(gen);
			// Get the spins associated with the bond
			s1 = b;
			s2 = (b + 1);
			if (b == n_bonds)
				s2 = 1; // Periodic boundary conditions
			if (spin[s1] != spin[s2])
			{
				if (dist(gen) * (M - nh) < J[b] * prob_insert)
				{
					op_string[i] = 2 * b;
					nh += 1;
				}
			}
		}
		// If there is a diagonal operator there, remove it
		else if (op % 2 == 0)
		{
			b = op / 2;
			if (dist(gen) * J[b] < prob_remove * (M - nh + 1))
			{
				op_string[i] = 0;
				nh -= 1;
			}
		}
		// If there is an off-diagonal operator, propagate
		else
		{
			b = op / 2;
			s1 = b;
			s2 = b + 1;
			if (b == n_bonds)
				s2 = 1; // Periodic boundary conditions
			// Flip the spins
			spin[s1] *= -1;
			spin[s2] *= -1;
		}
	}
}

void Heisenberg1D::CreateVertexList(void)
{
	int b, op, s1, s2, v1, v2;

	// Reset first_op and last_op to -1

	fill(first_op.begin(), first_op.end(), -1);
	fill(last_op.begin(), last_op.end(), -1);

	// Run over all blocks in vertex_list

	for (int v0 = 0; v0 < 4 * M; v0 += 4)
	{
		op = op_string[v0 / 4];
		if (op != 0)
		{
			b = op / 2;
			s1 = b;
			s2 = b + 1;
			if (b == n_bonds)
				s2 = 1;

			v1 = last_op[s1];
			v2 = last_op[s2];

			if (v1 != -1)
			{
				vertex_list[v1] = v0;
				vertex_list[v0] = v1;
			}
			else
				first_op[s1] = v0;

			if (v2 != -1)
			{
				vertex_list[v2] = v0 + 1;
				vertex_list[v0 + 1] = v2;
			}
			else
				first_op[s2] = v0 + 1;

			last_op[s1] = v0 + 2;
			last_op[s2] = v0 + 3;
		}
		else
		{
			for (int i = 0; i < 4; i++)
				vertex_list[v0 + i] = -1;
		}
	}

	// Create links across time boundary

	for (int s = 1; s <= L; s++)
	{
		v1 = first_op[s];
		if (v1 != -1)
		{
			v2 = last_op[s];
			vertex_list[v2] = v1;
			vertex_list[v1] = v2;
		}
	}
}

void Heisenberg1D::LoopUpdate(double beta, std::mt19937 &gen)
{

	int op, v1, v2;
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	// Traverse each loop and attemp to flip them (only marking)

	for (int v0 = 0; v0 < 4 * M; v0 += 2)
	{
		if (vertex_list[v0] < 0)
			continue;
		v1 = v0;
		// With probability 1/2, mark each cluster as visited and 'flipped' (-2)
		if (dist(gen) < 0.5)
		{
			while (true)
			{
				op = v1 / 4;
				op_string[op] = op_string[op] ^ 1; // XOR with 1, to flip the last bit: even <-> odd
				vertex_list[v1] = -2;
				v2 = v1 ^ 1;
				v1 = vertex_list[v2];
				vertex_list[v2] = -2;
				if (v1 == v0)
					break; // If the loop is closed, break
			}
		}
		// If not flipped, mark each cluster as visited only (-1)
		else
		{
			while (true)
			{
				vertex_list[v1] = -1;
				v2 = v1 ^ 1;
				v1 = vertex_list[v2];
				vertex_list[v2] = -1;
				if (v1 == v0)
					break;
			}
		}
	}

	// Flip the marked clusters and spins with no operators in worldline

	for (int s = 1; s <= L; s++)
	{
		v1 = first_op[s];
		// Flip the spins in the marked clusters
		if (v1 != -1)
		{
			if (vertex_list[v1] == -2)
				spin[s] *= -1;
		}
		// Attempt to flip the rest of the spins with no cluster
		else
		{
			if (dist(gen) < 0.5)
				spin[s] *= -1;
		}
	}
}

void Heisenberg1D::AdjustCutoff(void)
{

	int M_new = nh + nh / 3;
	if (M_new > M)
	{
		op_string.insert(op_string.end(), M_new - M, 0);
		vertex_list.insert(vertex_list.end(), 4 * (M_new - M), -1);
		M = M_new;
	}
}

void Heisenberg1D::DoubleSystem(void)
{
	// op_string.insert(op_string.end(), op_string.rbegin(), op_string.rend());
	op_string.insert(op_string.end(), op_string.begin(), op_string.end());
	vertex_list.insert(vertex_list.end(), 4 * M, -1);
	M += M;
	nh += nh;
}

void Heisenberg1D::shrink(void)
{
	first_op.shrink_to_fit();
	last_op.shrink_to_fit();
	op_string.shrink_to_fit();
	vertex_list.shrink_to_fit();
}

void Heisenberg1D::clear(void)
{
	first_op.clear();
	last_op.clear();
	op_string.clear();
	vertex_list.clear();
}

void Heisenberg1D::ComputeCorr(void)
{
	int b, op, s1, s2;
	int num_p = 1 + (M - 1) / L;
	double norm_factor = 1.0 / (2.0 * L * num_p);

	zz_corr.assign(L / 2, 0.0);

	std::vector<int> spin_copy;
	spin_copy.assign(spin.begin(), spin.end());

	for (int p = 0; p < M - 1; p++)
	{
		op = op_string[p];
		if (op != 0)
		{
			if (op % 2 == 1)
			{
				b = op / 2;
				s1 = b;
				s2 = b + 1;
				if (b == n_bonds)
					s2 = 1;
				spin_copy[s1] *= -1;
				spin_copy[s2] *= -1;
			}
		}
		if (p % L == 0)
		{
			for (int l = 0; l < L / 2; l++)
			{
				for (int i = 1; i <= L / 2; i++)
					zz_corr[l] += spin_copy[i] * spin_copy[i + 1 + l];
			}
		}
	}
	for (int l = 0; l < L / 2; l++)
		zz_corr[l] *= norm_factor;
}

double Heisenberg1D::ComputeCorrMid(void)
{
	int b, op, s1, s2;
	int num_p = 1 + (M - 1) / L;
	double zz_corr_mid = 0.0;

	std::vector<int> spin_copy;
	spin_copy.assign(spin.begin(), spin.end());

	for (int p = 0; p < M - 1; p++)
	{
		op = op_string[p];
		if (op != 0)
		{
			if (op % 2 == 1)
			{
				b = op / 2;
				s1 = b;
				s2 = b + 1;
				if (b == n_bonds)
					s2 = 1;
				spin_copy[s1] *= -1;
				spin_copy[s2] *= -1;
			}
		}
		if (p % L == 0)
		{
			for (int i = 1; i <= L / 2; i++)
				zz_corr_mid += spin_copy[i] * spin_copy[i + L / 2];
		}
	}
	zz_corr_mid *= 1.0 / (2.0 * L * num_p); // (1 / num_p) * (2 / L) * (1 / 4)
	return zz_corr_mid;
}

void Heisenberg1D::run(double beta, int MCS_eq, int MCS_sa, long int seed_eq)
{
	/*
	Runs MCS_eq equilibration sweeps followed by MCS_sa sampling sweeps, updates sample_zz_corr and run_nh
	*/
	double sumE = 0.0;
	
	sample_zz_corr.assign(L / 2, 0.0);
	run_nh.assign(MCS_sa, 0.0);

	std::mt19937 gen(seed_eq);

	for (int i = 0; i < MCS_eq; i++)
	{
		DiagonalUpdate(beta, gen);
		CreateVertexList();
		LoopUpdate(beta, gen);
		AdjustCutoff();
	}
  	shrink();

	for (int i = 0; i < MCS_sa; i++)
	{
		DiagonalUpdate(beta, gen);
		CreateVertexList();
		LoopUpdate(beta, gen);
		ComputeCorr();
		plus_equal_vector(sample_zz_corr, zz_corr);
		run_nh[i] = nh;
		sumE += nh;
	}

	scale_vector(sample_zz_corr, 1.0 / MCS_sa);

	E = sumE / MCS_sa;

	// Get energy per site

	E = -(E / beta - 0.25 * std::accumulate(J.begin()+1, J.end(), 0.0)) / L;

	// Free the memory
	clear();
	shrink();
}

void Heisenberg1D::equilibrium(double beta, int MCS_run_eq, long int seed_eq)
{
	run_zz_corr_mid.assign(MCS_run_eq, 0.0);
	run_nh.assign(MCS_run_eq, 0.0);

	std::mt19937 gen(seed_eq);

	for(int t = 0; t < MCS_run_eq; t++)
	{
		DiagonalUpdate(beta, gen);
		CreateVertexList();
		LoopUpdate(beta, gen);
		run_zz_corr_mid[t] = ComputeCorrMid();
		run_nh[t] = nh;
		AdjustCutoff();
	}
}

void Heisenberg1D::beta_doubling(int n_max, int MCS_eq, long int seed_eq)
{
	double sumE = 0.0;

	double beta = 1.0;

	sample_zz_corr.assign(L / 2, 0.0);

	sample_zz_corr_mid.assign(n_max+1, 0.0);

	double zz_corr_mid = 0.0;

	std::mt19937 gen(seed_eq);

	for (int n = 0; n < n_max; n++)
	{
		for (int i = 0; i < MCS_eq; i++)
		{
			DiagonalUpdate(beta, gen);
			CreateVertexList();
			LoopUpdate(beta, gen);
			AdjustCutoff();
		}
		shrink();
		for (int i = 0; i < 2 * MCS_eq; i++)
		{
			DiagonalUpdate(beta, gen);
			CreateVertexList();
			LoopUpdate(beta, gen);
			sample_zz_corr_mid[n] += ComputeCorrMid();
		}
		sample_zz_corr_mid[n] /= 2 * MCS_eq;
		DoubleSystem();
		beta *= 2;
	}

	// Sample the final beta

	for (int i = 0; i < MCS_eq; i++) // Since this is the final beta, we give it more iterations 
	{
		DiagonalUpdate(beta, gen);
		CreateVertexList();
		LoopUpdate(beta, gen);
		AdjustCutoff();
	}
	shrink();
	for (int i = 0; i < 2 * MCS_eq; i++)
	{
		DiagonalUpdate(beta, gen);
		CreateVertexList();
		LoopUpdate(beta, gen);
		ComputeCorr();
		sample_zz_corr_mid[n_max] += ComputeCorrMid();
		plus_equal_vector(sample_zz_corr, zz_corr);
		sumE += nh;
	}
	sample_zz_corr_mid[n_max] /= 2 * MCS_eq;

	scale_vector(sample_zz_corr, 1.0 / (2 * MCS_eq));

	E = sumE / (2 * MCS_eq);

	// Get energy per site

	E = -(E / beta - 0.25 * std::accumulate(J.begin(), J.end(), 0.0)) / L;

	// Free the memory
	clear();
	shrink();
}
