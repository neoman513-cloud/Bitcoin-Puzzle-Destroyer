// author: https://t.me/biernus
#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <fstream>
#include <stdint.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <inttypes.h>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(__NT__)
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#elif defined(__linux__)
#include <sys/random.h>
#endif
#include <chrono>
#pragma once

__device__ __host__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

// Convert hex string to bytes
__device__ __host__ __device__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}


// Convert hex string to BigInt - optimized
__device__ __host__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
    // Process hex string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Convert BigInt to hex string - optimized
__device__ __host__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

// Optimized byte to hex conversion
__device__ __host__ __forceinline__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ __host__ void hash160_to_hex(uint8_t* hash, char* hex_str) {
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        byte_to_hex(hash[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}


__device__ __forceinline__ bool compare_hash160_fast(const uint8_t* hash1, const uint8_t* hash2) {
    uint64_t a1, a2, b1, b2;
    uint32_t c1, c2;
    
    memcpy(&a1, hash1, 8);
    memcpy(&a2, hash1 + 8, 8);
    memcpy(&c1, hash1 + 16, 4);

    memcpy(&b1, hash2, 8);
    memcpy(&b2, hash2 + 8, 8);
    memcpy(&c2, hash2 + 16, 4);

    return (a1 == b1) && (a2 == b2) && (c1 == c2);
}

__device__ void hash160_to_hex(const uint8_t *hash, char *out_hex) {
    const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 20; ++i) {
        out_hex[i * 2]     = hex_chars[hash[i] >> 4];
        out_hex[i * 2 + 1] = hex_chars[hash[i] & 0x0F];
    }
    out_hex[40] = '\0';
}

// Device function to generate random BigInt in range [min, max]
__device__ void generate_random_in_range(BigInt* result, curandStatePhilox4_32_10_t* state, 
                                         const BigInt* min_val, const BigInt* max_val) {
    // Calculate range = max - min
    BigInt range;
    bool borrow = false;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t diff = (uint64_t)max_val->data[i] - (uint64_t)min_val->data[i] - (borrow ? 1 : 0);
        range.data[i] = (uint32_t)diff;
        borrow = (diff > 0xFFFFFFFFULL);
    }
    
    // Generate random value in [0, range]
    BigInt random;
    for (int w = 0; w < BIGINT_WORDS; w += 4) {
        uint4 r = curand4(state);
        if (w + 0 < BIGINT_WORDS) random.data[w + 0] = r.x;
        if (w + 1 < BIGINT_WORDS) random.data[w + 1] = r.y;
        if (w + 2 < BIGINT_WORDS) random.data[w + 2] = r.z;
        if (w + 3 < BIGINT_WORDS) random.data[w + 3] = r.w;
    }
    
    // Reduce random to range using modulo-like operation
    // Simple approach: if random > range, regenerate or use rejection sampling
    // For efficiency, we'll use: random = random % (range + 1)
    // But since we don't have full bigint modulo, we'll use a simpler approach:
    // Keep only the significant bits based on range's highest bit
    
    // Find highest set bit in range
    int highest_word = BIGINT_WORDS - 1;
    while (highest_word >= 0 && range.data[highest_word] == 0) {
        highest_word--;
    }
    
    if (highest_word >= 0) {
        // Mask off bits beyond the range
        uint32_t mask = range.data[highest_word];
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        
        random.data[highest_word] &= mask;
        
        // Zero out higher words
        for (int i = highest_word + 1; i < BIGINT_WORDS; ++i) {
            random.data[i] = 0;
        }
        
        // Check if random <= range, if not, reduce it
        bool greater = false;
        for (int i = BIGINT_WORDS - 1; i >= 0; --i) {
            if (random.data[i] > range.data[i]) {
                greater = true;
                break;
            } else if (random.data[i] < range.data[i]) {
                break;
            }
        }
        
        if (greater) {
            // Simple reduction: random = random & (mask >> 1)
            for (int i = 0; i < BIGINT_WORDS; ++i) {
                random.data[i] = random.data[i] % (range.data[i] + 1);
            }
        }
    }
    
    // Add min: result = random + min
    bool carry = false;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t sum = (uint64_t)random.data[i] + (uint64_t)min_val->data[i] + (carry ? 1 : 0);
        result->data[i] = (uint32_t)sum;
        carry = (sum > 0xFFFFFFFFULL);
    }
}

__constant__ BigInt d_min_bigint;
__constant__ BigInt d_max_bigint;

__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};
__device__ char g_found_hash160[41] = {0};

__global__ void start(const uint8_t* target, uint64_t p1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize RNG once
    curandStatePhilox4_32_10_t state;
    curand_init(p1, tid, 0, &state);
    
    ECPointJac result_jac_batch[BATCH_SIZE];
    BigInt priv_batch[BATCH_SIZE];
    uint8_t hash160_batch[BATCH_SIZE][20];
    
	#pragma unroll
	for (int i = 0; i < BATCH_SIZE; ++i) {
		generate_random_in_range(&priv_batch[i], &state, &d_min_bigint, &d_max_bigint);
		scalar_multiply_multi_base_jac(&result_jac_batch[i], &priv_batch[i]);
	}
	
	jacobian_batch_to_hash160(result_jac_batch, hash160_batch);
	
	#pragma unroll
	for (int i = 0; i < BATCH_SIZE; ++i) {
		if (compare_hash160_fast(hash160_batch[i], target)) {
			if (atomicCAS((int*)&g_found, 0, 1) == 0) {
				bigint_to_hex(&priv_batch[i], g_found_hex);
				hash160_to_hex(hash160_batch[i], g_found_hash160);
			}
			return;
		}
	}
}

bool run_with_quantum_data(const char* min, const char* max, const char* target, int blocks, int threads, int device_id) {
    uint8_t shared_target[20];
    hex_string_to_bytes(target, shared_target, 20);
    uint8_t *d_target;
    cudaMalloc(&d_target, 20);
    cudaMemcpy(d_target, shared_target, 20, cudaMemcpyHostToDevice);
    
    // Convert min and max hex strings to BigInt and copy to device
    BigInt min_bigint, max_bigint;
    hex_to_bigint(min, &min_bigint);
    hex_to_bigint(max, &max_bigint);
    
    cudaMemcpyToSymbol(d_min_bigint, &min_bigint, sizeof(BigInt));
    cudaMemcpyToSymbol(d_max_bigint, &max_bigint, sizeof(BigInt));
    
    int total_threads = blocks * threads;
    int found_flag;
    
    // Calculate keys processed per kernel launch
    uint64_t keys_per_kernel = (uint64_t)blocks * threads * BATCH_SIZE;
    
    printf("Searching in range:\n");
    printf("Min: %s\n", min);
    printf("Max: %s\n", max);
    printf("Target: %s\n", target);
    printf("Blocks: %d, Threads: %d, Batch size: %d\n", blocks, threads, BATCH_SIZE);
    printf("Total threads: %d\n", total_threads);
    printf("Keys per kernel: %llu\n\n", (unsigned long long)keys_per_kernel);
    
    uint64_t p1;
    // Performance tracking variables
    uint64_t total_keys_checked = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print_time = start_time;
        #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(__NT__)
	BCryptGenRandom(NULL, (PUCHAR)&p1, sizeof(p1), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
        #elif defined(__linux__)
        getrandom((unsigned char *)&p1, sizeof(p1), GRND_RANDOM);
        #endif
    while(true) {
        auto kernel_start = std::chrono::high_resolution_clock::now();
        
        // Launch kernel
        start<<<blocks, threads>>>(d_target, p1);
        cudaDeviceSynchronize();
        
        auto kernel_end = std::chrono::high_resolution_clock::now();
        
        // Calculate kernel execution time
        double kernel_time = std::chrono::duration<double>(kernel_end - kernel_start).count();
        
        // Update counters
        total_keys_checked += keys_per_kernel;
        
        // Print performance stats every second
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_since_print = std::chrono::duration<double>(current_time - last_print_time).count();
        
        if (elapsed_since_print >= 1.0) {
            double current_kps = keys_per_kernel / kernel_time;
            
            printf("\rSpeed: %.2f MK/s | Total: %.2f B keys",
                   current_kps / 1000000.0,
                   total_keys_checked / 1000000000.0);
            fflush(stdout);
            
            last_print_time = current_time;
        }
        
        // Check if key was found
        cudaMemcpyFromSymbol(&found_flag, g_found, sizeof(int));
        if (found_flag) {
            printf("\n\n");
            
            char found_hex[65], found_hash160[41];
            cudaMemcpyFromSymbol(found_hex, g_found_hex, 65);
            cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41);
            
            double total_time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start_time
            ).count();
            
            printf("FOUND!\n");
            printf("Private Key: %s\n", found_hex);
            printf("Hash160: %s\n", found_hash160);
            printf("Total time: %.2f seconds\n", total_time);
            printf("Total keys checked: %llu (%.2f billion)\n", 
                   (unsigned long long)total_keys_checked,
                   total_keys_checked / 1000000000.0);
            printf("Average speed: %.2f MK/s\n", total_keys_checked / total_time / 1000000.0);
            
            std::ofstream outfile("result.txt", std::ios::app);
            if (outfile.is_open()) {
                std::time_t now = std::time(nullptr);
                char timestamp[100];
                std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                outfile << "[" << timestamp << "] Found: " << found_hex << " -> " << found_hash160 << std::endl;
                outfile << "Total keys checked: " << total_keys_checked << std::endl;
                outfile << "Time taken: " << total_time << " seconds" << std::endl;
                outfile << "Average speed: " << (total_keys_checked / total_time / 1000000.0) << " MK/s" << std::endl;
                outfile << std::endl;
                outfile.close();
                std::cout << "Result appended to result.txt" << std::endl;
            }
            
            cudaFree(d_target);
            return true;
        }
        p1 += 1;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <min> <max> <target> [device_id]" << std::endl;
        return 1;
    }
    int blocks = 4096;
    int threads = 256;
    int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;
    
    // Set GPU device
    cudaSetDevice(device_id);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error setting device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Validate input lengths match
    if (strlen(argv[1]) != strlen(argv[2])) {
        std::cerr << "Error: min and max must have the same length" << std::endl;
        return 1;
    }
    init_gpu_constants();
    cudaDeviceSynchronize();
    bool result = run_with_quantum_data(argv[1], argv[2], argv[3], blocks, threads, device_id);
    
    return 0;
}
