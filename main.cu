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
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#pragma once

__device__ __host__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

__device__ __host__ __device__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

__device__ __host__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
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

__device__ __host__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
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
    
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

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
#include <cuda_runtime.h> 



__device__ bool is_zero_bigint(const BigInt* num) {
    bool is_zero = true;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        if (num->data[i] != 0) {
            is_zero = false;
            
            
        }
    }
    return is_zero;
}

__device__ void generate_random_in_range(BigInt* result, curandStatePhilox4_32_10_t* state, 
                                         const BigInt* min_val, const BigInt* max_val) {
    
    BigInt range;
    bool borrow = false;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t diff = (uint64_t)max_val->data[i] - (uint64_t)min_val->data[i] - (borrow ? 1 : 0);
        range.data[i] = (uint32_t)diff;
        borrow = (diff > 0xFFFFFFFFULL);
    }

    
    
    
    if (is_zero_bigint(&range)) {
        #pragma unroll
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            result->data[i] = min_val->data[i];
        }
        return;
    }
    
    
    
    
    int highest_word_idx = BIGINT_WORDS - 1;
    #pragma unroll
    for (int i = BIGINT_WORDS - 1; i >= 0; --i) {
        
        if (range.data[i] != 0) {
            highest_word_idx = i;
            break;
        }
    }

    
    
    uint32_t mask = range.data[highest_word_idx];
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;

    BigInt random;
    
    
    do {
        
        for (int w = 0; w < BIGINT_WORDS; w += 4) {
            uint4 r = curand4(state);
            if (w + 0 < BIGINT_WORDS) random.data[w + 0] = r.x;
            if (w + 1 < BIGINT_WORDS) random.data[w + 1] = r.y;
            if (w + 2 < BIGINT_WORDS) random.data[w + 2] = r.z;
            if (w + 3 < BIGINT_WORDS) random.data[w + 3] = r.w;
        }

        
        
        
        #pragma unroll
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            
            if (i > highest_word_idx) {
                random.data[i] = 0;
            } else if (i == highest_word_idx) {
                random.data[i] &= mask;
            }
        }

    
    
    } while (compare_bigint(&random, &range) > 0);

    
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

struct UltraCompactResult {
    uint32_t hash160[5];  
};


__global__ void start_store_keys(uint64_t p1, UltraCompactResult* d_results, BigInt* d_privkeys)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int global_offset = tid * BATCH_SIZE;
    
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
        int result_idx = global_offset + i;
        
        const uint32_t* src = (const uint32_t*)hash160_batch[i];
        uint32_t* dst = d_results[result_idx].hash160;
        dst[0] = src[0]; 
        dst[1] = src[1]; 
        dst[2] = src[2]; 
        dst[3] = src[3]; 
        dst[4] = src[4];
        
        d_privkeys[result_idx] = priv_batch[i];
    }
}


std::atomic<bool> g_found(false);
std::mutex g_print_mutex;


struct FoundKey {
    char privkey_hex[65];
    char hash160_hex[41];
    uint64_t keys_checked;
    double time_elapsed;
};


inline bool compare_hash160_simd(const uint8_t* hash1, const uint8_t* hash2) {
#ifdef __AVX2__
    __m256i h1 = _mm256_loadu_si256((__m256i*)hash1);
    __m256i h2 = _mm256_loadu_si256((__m256i*)hash2);
    __m256i cmp = _mm256_cmpeq_epi8(h1, h2);
    int mask = _mm256_movemask_epi8(cmp);
    return (mask & 0xFFFFF) == 0xFFFFF;
#else
    
    const uint64_t* h1_64 = (const uint64_t*)hash1;
    const uint64_t* h2_64 = (const uint64_t*)hash2;
    const uint32_t* h1_32 = (const uint32_t*)hash1;
    const uint32_t* h2_32 = (const uint32_t*)hash2;
    
    return (h1_64[0] == h2_64[0]) && 
           (h1_64[1] == h2_64[1]) && 
           (h1_32[4] == h2_32[4]);
#endif
}

void check_results_worker(
    const UltraCompactResult* h_results,
    const uint8_t* target_hash,
    uint64_t start_idx,
    uint64_t end_idx,
    uint64_t* found_index,
    std::atomic<bool>* local_found)
{
    
    const uint32_t* target_u32 = (const uint32_t*)target_hash;
    
    for (uint64_t i = start_idx; i < end_idx; ++i) {
        if (local_found->load(std::memory_order_relaxed)) return;
        
        const uint32_t* hash = h_results[i].hash160;
        
        
        if (hash[0] != target_u32[0]) continue;
        if (hash[1] != target_u32[1]) continue;
        if (hash[2] != target_u32[2]) continue;
        if (hash[3] != target_u32[3]) continue;
        if (hash[4] != target_u32[4]) continue;
        
        if (!local_found->exchange(true, std::memory_order_relaxed)) {
            *found_index = i;
        }
        return;
    }
}


bool run_with_quantum_data(const char* min, const char* max, const char* target, int blocks, int threads, int device_id) {
    uint8_t target_hash[20];
    hex_string_to_bytes(target, target_hash, 20);
    
    BigInt min_bigint, max_bigint;
    hex_to_bigint(min, &min_bigint);
    hex_to_bigint(max, &max_bigint);
    
    cudaMemcpyToSymbol(d_min_bigint, &min_bigint, sizeof(BigInt));
    cudaMemcpyToSymbol(d_max_bigint, &max_bigint, sizeof(BigInt));
    
    int total_threads = blocks * threads;
    uint64_t results_per_kernel = (uint64_t)total_threads * BATCH_SIZE;
    
    
    const int NUM_BUFFERS = 3;
    cudaStream_t streams[NUM_BUFFERS];
    
    
    UltraCompactResult* d_results[NUM_BUFFERS]; 
    UltraCompactResult* h_results[NUM_BUFFERS];
    BigInt* d_privkeys[NUM_BUFFERS];            
    
    
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cudaStreamCreate(&streams[i]);
        
        
        cudaMalloc(&d_results[i], results_per_kernel * sizeof(UltraCompactResult));
        cudaMallocHost(&h_results[i], results_per_kernel * sizeof(UltraCompactResult));
        
        
        cudaMalloc(&d_privkeys[i], results_per_kernel * sizeof(BigInt));
    }
    
    unsigned int num_cpu_threads = std::thread::hardware_concurrency();
    if (num_cpu_threads == 0) num_cpu_threads = 4;
    num_cpu_threads = std::min(num_cpu_threads * 2, 32u);
    
    
    printf("=== OPTIMIZED DATA TRANSFER MODE ===\n");
    printf("Searching in range: %s to %s\n", min, max);
    printf("Target: %s\n", target);
    printf("Blocks: %d, Threads: %d, Batch size: %d\n", blocks, threads, BATCH_SIZE);
    printf("CPU threads: %u\n", num_cpu_threads);
    printf("Results per kernel: %llu\n", (unsigned long long)results_per_kernel);
    printf("Buffers: %d streams with pinned memory\n", NUM_BUFFERS);
    
    printf("Hash160 buffer: %.2f MB per stream (TRANSFERRED)\n", 
           (results_per_kernel * sizeof(UltraCompactResult)) / (1024.0 * 1024.0));
    printf("PrivKey buffer: %.2f MB per stream (ON-DEVICE ONLY)\n", 
           (results_per_kernel * sizeof(BigInt)) / (1024.0 * 1024.0));
    printf("Data transferred per stream: %.2f MB (from ~1.7GB down to %.2fMB)!\n\n", 
           (results_per_kernel * (sizeof(UltraCompactResult) + sizeof(BigInt))) / (1024.0 * 1024.0),
           (results_per_kernel * sizeof(UltraCompactResult)) / (1024.0 * 1024.0)
    );
    
    uint64_t p1;
    uint64_t total_keys_checked = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print_time = start_time;
    
    
    std::random_device rd;
    std::mt19937_64 gen(rd());
    p1 = gen();
    
    g_found.store(false);
    
    int iteration = 0;
    
    
    start_store_keys<<<blocks, threads, 0, streams[0]>>>(p1, d_results[0], d_privkeys[0]);
    start_store_keys<<<blocks, threads, 0, streams[1]>>>(p1 + 1, d_results[1], d_privkeys[1]);
    
    while(true) {
        int current_buffer = iteration % NUM_BUFFERS;
        int next_buffer = (iteration + 1) % NUM_BUFFERS;
        
        
        start_store_keys<<<blocks, threads, 0, streams[next_buffer]>>>(p1 + iteration + NUM_BUFFERS - 1, d_results[next_buffer], d_privkeys[next_buffer]);
        
        
        cudaMemcpyAsync(h_results[current_buffer], d_results[current_buffer], 
                        results_per_kernel * sizeof(UltraCompactResult), 
                        cudaMemcpyDeviceToHost, streams[current_buffer]);
        
        
        cudaStreamSynchronize(streams[current_buffer]);
        
        
        std::vector<std::thread> check_threads;
        std::atomic<bool> local_found(false);
        uint64_t local_found_index = 0;
        
        uint64_t chunk_size = results_per_kernel / num_cpu_threads;
        
        for (unsigned int t = 0; t < num_cpu_threads; ++t) {
            uint64_t start_idx = t * chunk_size;
            uint64_t end_idx = (t == num_cpu_threads - 1) ? results_per_kernel : (t + 1) * chunk_size;
            
            check_threads.emplace_back(check_results_worker, h_results[current_buffer], target_hash, start_idx, end_idx, &local_found_index, &local_found);
        }
        
        for (auto& th : check_threads) {
            th.join();
        }
        
        
        if (local_found.load(std::memory_order_seq_cst)) {
            g_found.store(true);

            printf("\n\n=== MATCH FOUND by CPU worker! ===\n");
            printf("Retrieving private key from device...\n");

            
            BigInt found_privkey;

            
            cudaMemcpy(&found_privkey, &d_privkeys[current_buffer][local_found_index], sizeof(BigInt), cudaMemcpyDeviceToHost);
            
            char privkey_hex[65];
            char hash160_hex[41];
			
			uint8_t found_hash_bytes[20];
			memcpy(found_hash_bytes, h_results[current_buffer][local_found_index].hash160, 20);
			bigint_to_hex(&found_privkey, privkey_hex);
			hash160_to_hex(found_hash_bytes, hash160_hex);
            auto now = std::chrono::high_resolution_clock::now();
            double found_time = std::chrono::duration<double>(now - start_time).count();
            uint64_t total_checked = total_keys_checked + local_found_index;
            
            printf("\n*** FOUND! ***\n");
            printf("Private Key: %s\n", privkey_hex);
            printf("Hash160: %s\n", hash160_hex);
            printf("Total time: %.2f seconds\n", found_time);
            printf("Total keys checked: %llu (%.2f billion)\n", (unsigned long long)total_checked, total_checked / 1000000000.0);
            printf("Average speed: %.2f MK/s\n", total_checked / found_time / 1000000.0);

            
            std::ofstream outfile("result.txt", std::ios::app);
            if (outfile.is_open()) {
                std::time_t t = std::time(nullptr);
                char timestamp[100];
                std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
                outfile << "[" << timestamp << "] Found: " << privkey_hex << " -> " << hash160_hex << std::endl;
                outfile << "Total keys checked: " << total_checked << std::endl;
                outfile << "Time taken: " << found_time << " seconds" << std::endl;
                outfile << "Average speed: " << (total_checked / found_time / 1000000.0) << " MK/s" << std::endl;
                outfile << std::endl;
                outfile.close();
            }
            
            
            for (int i = 0; i < NUM_BUFFERS; i++) {
                cudaFreeHost(h_results[i]);
                cudaFree(d_privkeys[i]); 
                cudaFree(d_results[i]);
                cudaStreamDestroy(streams[i]);
            }
            return true; 
        }
        
        total_keys_checked += results_per_kernel;
        
        
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_since_print = std::chrono::duration<double>(current_time - last_print_time).count();
        if (elapsed_since_print >= 0.5) {
            double total_elapsed = std::chrono::duration<double>(current_time - start_time).count();
            double current_speed = (total_keys_checked / total_elapsed) / 1000000.0;
            printf("\r[%.1fs] Speed: %.2f MK/s | Total: %.3f B keys        ",
                   total_elapsed, current_speed, total_keys_checked / 1000000000.0);
            fflush(stdout);
            last_print_time = current_time;
        }
        
        iteration++;
    }
    
    
    /*
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cudaFreeHost(h_results[i]);
        cudaFree(d_results[i]);
        cudaFree(d_privkeys[i]);
        cudaStreamDestroy(streams[i]);
    }
    */
    return false;
}
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <min> <max> <target> [device_id]" << std::endl;
        return 1;
    }
    int blocks = 256;
    int threads = 256;
    int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;
    
    cudaSetDevice(device_id);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error setting device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    if (strlen(argv[1]) != strlen(argv[2])) {
        std::cerr << "Error: min and max must have the same length" << std::endl;
        return 1;
    }
    
    init_gpu_constants();
    cudaDeviceSynchronize();
    bool result = run_with_quantum_data(argv[1], argv[2], argv[3], blocks, threads, device_id);
    
    return 0;
}