#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <math.h>

using namespace std;
using namespace std::chrono;

#define N 100				// default number of investigated nodes (first N ID's of the network)
#define THRESHOLD 0.5		// default nodes' threshold value
#define THREADNUM 192		// default number of threads within blocks
#define testRuns 1			// default number of test runs. Change in order to measure the average time for a few test runs.
#define ITERATIONS 10000	// default maximum number of iterations

__global__ void warmUp()	// GPU warm-up before main calculations for better, stable results
{
	int x = 0;
	for (int i = 0; i < 1000; i++)
	{
		x++;
	}
}

__global__ void CountInfluence(bool *a, float *b, float *c)
{
	float temp = 0;

	int id = blockIdx.x * blockDim.x + threadIdx.x; // unique ID number

	if (id < N)
	{
		for (int i = 0; i < N; i++)
		{
			if (id != i)
			{
				temp += a[i] * b[i * N + id];
			}
		}
		c[id] = temp;
	}
}

int readFile(float inf_f[][N], int connections_i[])
{
	const int MAX_CHARS_PER_LINE = 512;
	const int MAX_TOKENS_PER_LINE = 20;
	const char* const DELIMITER = " ";

	string dataset_path;
	string dataset_number_input;
	int dataset_number;

	cout << endl << "Which dataset do you want to use? Type '1' '2' or '3'" << endl;
	cout << "[1] UC Irvine messages" << endl;
	cout << "[2] Digg" << endl;
	cout << "[3] Facebook wall posts" << endl;

	getline(cin, dataset_number_input);
	stringstream(dataset_number_input) >> dataset_number;

	switch (dataset_number)
	{
		case 1:
		{
			dataset_path = "datasets/opsahl-ucforum/out.opsahl-ucforum";
			break;
		}
		case 2:
		{
			dataset_path = "datasets/munmun_digg_reply/out.munmun_digg_reply";
			break;
		}
		case 3:
		{
			dataset_path = "datasets/facebook-wosn-wall/out.facebook-wosn-wall";
			break;
		}
		default:
		{
			cout << "Wrong number, try again." << endl;
			return 1;
		}
	}


	ifstream fin;
	fin.open(dataset_path);

	if (!fin.good())
	{
		cout << "File not found.";
		return 1;
	}

	while (!fin.eof())
	{
		char buf[MAX_CHARS_PER_LINE];
		fin.getline(buf, MAX_CHARS_PER_LINE);

		int n = 0;
		const char* token[MAX_TOKENS_PER_LINE] = {};

		token[0] = strtok(buf, DELIMITER);
		if (token[0])
		{
			for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
			{
				token[n] = strtok(0, DELIMITER);
				if (!token[n]) break;
			}
			if (atoi(token[0]) - 1 < N && atoi(token[1]) - 1 < N)
			{
				inf_f[atoi(token[0]) - 1][atoi(token[1]) - 1] += 1;  // calculating the total number of iteractions from "i" to "j"
				connections_i[atoi(token[1]) - 1] += 1; // calculating the total number of received interactions by the "j" node
			}
		}
	}

	for (int i = 0; i < N; i++) // Influence value calculated as the ratio of iteractions from "i" node to "j" node, to the total number of received iteractions by the "j" node.
	{
		for (int j = 0; j < N; j++)
		{
			if (connections_i[i] != 0)
			{
				inf_f[i][j] = inf_f[i][j] / connections_i[j];
			}
		}
	}
	return 0;
}


int main()
{
	float inf_f[N][N];
	bool state_b[N];
	int state_changes_i[N];
	int connections_i[N];
	float result_f[N];

	// Setting the inital values
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			inf_f[i][j] = 0;
		}
		connections_i[i] = 0;
		state_b[i] = false;
		state_changes_i[i] = 0;
		result_f[i] = 0;
	}

	string input_toFind;
	int percToFind;

	cout << endl << "What is the percentage of all network nodes, that should be chosen as the best initially activated nodes (seeds)? Type the percentage value only." << endl;
	getline(cin, input_toFind);
	stringstream(input_toFind) >> percToFind;


	if (readFile(inf_f, connections_i))
	{
		cout << "Error reading file." << endl;
		return 1;
	}

	int toFind = (int)ceil(float(percToFind * N) / 100);

	warmUp << <1024, 1024 >> >();

	float timeAllRuns = 0;
	for (int t = 0; t < testRuns; t++)
	{
		for (int i = 0; i < N; i++)
		{
			state_b[i] = false;
		}

		high_resolution_clock::time_point beginning = high_resolution_clock::now();


		// Allocating memory for GPU matrices
		bool *d_state_temp_b;
		float *d_inf_f;
		float *d_result_f;

		if (cudaMalloc(&d_state_temp_b, sizeof(bool)* N) != cudaSuccess)
		{
			cout << "Error allocating memory for d_state_temp_b." << endl;
			return 1;
		}
		if (cudaMalloc(&d_inf_f, sizeof(float)* N *N) != cudaSuccess)
		{
			cout << "Error allocating memory for d_inf_f." << endl;
			cudaFree(d_state_temp_b);
			return 1;
		}
		if (cudaMalloc(&d_result_f, sizeof(float)* N) != cudaSuccess)
		{
			cout << "Error allocating memory for d_result_f." << endl;
			cudaFree(d_state_temp_b); cudaFree(d_inf_f);
			return 1;
		}


		// Copying initial values from Host to Device
		if (cudaMemcpy(d_inf_f, inf_f, sizeof(float)* N *N, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			cout << "Error copying inf_f to d_inf_f." << endl;
			cudaFree(d_state_temp_b); cudaFree(d_inf_f); cudaFree(d_result_f);
			delete[] inf_f;
			return 1;
		}


		for (int found = 0; found < toFind;)
		{
			for (int i = 0; i < N; i++)
			{
				state_changes_i[i] = 0;
			}

			for (int curr_i = 0; curr_i < N; curr_i++)
			{
				if (!state_b[curr_i])
				{
					bool state_temp_b[N];

					for (int el_st = 0; el_st < N; el_st++)
					{
						state_temp_b[el_st] = state_b[el_st];
					}

					state_temp_b[curr_i] = true;

					int changes_before_i = -1;
					int changes_after_i = 0;
					int it = 0;
					while (it < ITERATIONS && changes_before_i < changes_after_i)
					{
						changes_before_i = state_changes_i[curr_i];

						// Copy values from Host to Device
						if (cudaMemcpy(d_state_temp_b, state_temp_b, sizeof(bool)* N, cudaMemcpyHostToDevice) != cudaSuccess)
						{
							cout << "Error copying state_temp_b to d_state_temp_b." << endl;
							cudaFree(d_state_temp_b); cudaFree(d_inf_f); cudaFree(d_result_f);
							delete[] state_temp_b;
							return 1;
						}

						// GPU function called from CPU (blocks,threads)
						CountInfluence << <N / THREADNUM + 1, THREADNUM >> >(d_state_temp_b, d_inf_f, d_result_f);

						// Copy results from GPU to Host
						if (cudaMemcpy(result_f, d_result_f, sizeof(float)* N, cudaMemcpyDeviceToHost) != cudaSuccess)
						{
							cudaFree(d_state_temp_b); cudaFree(d_inf_f); cudaFree(d_result_f);
							delete[] state_temp_b; delete[] result_f;
							cout << "Error copying d_result_f to result_f" << endl;
							return 1;
						}

						for (int i = 0; i < N; i++)
						{
							if (result_f[i] >= THRESHOLD && !state_temp_b[i])
							{
								state_changes_i[curr_i]++;
								state_temp_b[i] = true;
							}
							result_f[i] = 0;
						}

						changes_after_i = state_changes_i[curr_i];
						it++;
					} // while (it < ITERATIONS && changes_before_i < changes_after_i) 

					state_temp_b[curr_i] = false;
				} // if (!state_b[curr_i])
			} // for (int curr_i = 0; curr_i < N; curr_i++)

			int maxactValue = 0;
			int maxactIndex = -1;

			for (int i = 0; i < N; i++)
			{
				if (state_changes_i[i] > maxactValue && !state_b[i])
				{
					maxactIndex = i;
					maxactValue = state_changes_i[i];
				}
			}

			state_b[maxactIndex] = true;

			found++;

			cout << endl << "Found: " << found << "/" << toFind;
			cout << endl << "Number of influenced: " << maxactValue << endl;
		}

		high_resolution_clock::time_point ending = high_resolution_clock::now();

		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(ending - beginning).count();

		cout << endl << endl << "Activated nodes: ";
		for (int i = 0; i < N; i++)
		{
			if (state_b[i])
			{
				cout << i << ", ";
			}
		}

		cout << endl << endl << "Execution time: " << duration / 1000 << "s." << endl;

		timeAllRuns += (duration / 1000);
	}

	cout << "Average execution time: " << timeAllRuns / testRuns << "s." << endl << endl;

	system("pause");
	return 0;
}
