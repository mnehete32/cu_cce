#ifndef UTILS
#define UTILS

#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

template <typename T> int fill_ptr(std::string filename, T * ptr, int64_t N) {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    std::cout << "Opening file: " << filename << std::endl;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file\n";
        return 1;
    }
    int64_t     count = 0;
    std::string line;

    while (std::getline(file, line) && count < N) {
        std::stringstream ss(line);
        float             tmp;
        while (ss >> tmp && count < N) {
            ptr[count++] = static_cast<T>(tmp);
        }
    }

    if (count < N) {
        std::cerr << "Warning: Only read " << count << " numbers (less than " << N << ")\n";
        return 1;
    }

    return 0;
}

#endif
