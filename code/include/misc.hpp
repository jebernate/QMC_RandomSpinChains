#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iterator>
#include <algorithm>
#include <functional>

void elapsed(int counter, std::chrono::time_point<std::chrono::steady_clock> const &start)
{
    const auto time = std::chrono::steady_clock::now() - start;
    const auto hrs = std::chrono::duration_cast<std::chrono::hours>(time);
    const auto mins = std::chrono::duration_cast<std::chrono::minutes>(time - hrs);
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(time - hrs - mins);

    const auto total_secs = std::chrono::duration_cast<std::chrono::seconds>(time);

    std::cout << "[" << std::right << std::setw(2) << std::setfill('0') << hrs.count() << ":";
    std::cout << std::right << std::setw(2) << std::setfill('0') << mins.count() << ":";
    std::cout << std::right << std::setw(2) << std::setfill('0') << secs.count() << ", ";
    std::cout << std::setprecision(1) << std::fixed << total_secs.count() / (1.0 * counter) << " s/it] \n";
}

void saveVector(const std::vector<double> &vector, std::string filename, bool append = true)
{
    // Open the file for writing
    std::ofstream outputFile;
    if (append)
        outputFile.open(filename, std::ios_base::app);

    else
        outputFile.open(filename);

    if (outputFile.is_open())
    {

        if (append)
            outputFile << "\n";

        // outputFile << std::setprecision(8);

        // Define the iterator
        std::ostream_iterator<double> outputIterator(outputFile, ",");

        // Use a fold expression to iterate over all vectors and save their contents to the file
        std::copy(vector.begin(), vector.end() - 1, outputIterator);
        outputFile << vector.back(); // Last element without a comma at the end

        // Close the file
        outputFile.close();
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void plus_equal_vector(std::vector<double> &a, const std::vector<double> &b)
{
    // Sum a and b and store the result in a (i.e a += b)
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<double>());
}

void scale_vector(std::vector<double> &v, double c)
{
    // Perform multiplication by scalar
    std::transform(v.begin(), v.end(), v.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, c));
}

void sum_square(std::vector<double> &accumulator, const std::vector<double> &vector)
{
    for (int i = 0; i < vector.size(); i++)
        accumulator[i] += vector[i] * vector[i];
}
