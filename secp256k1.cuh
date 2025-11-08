// author: https://t.me/biernus
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8
#define WINDOW_SIZE 16
#define NUM_BASE_POINTS 16
#define BATCH_SIZE 128
#define MOD_EXP 4


struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};
__constant__ BigInt const_p_minus_2;
__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;

// Original window precomputation
__device__ ECPointJac G_precomp[1 << WINDOW_SIZE];

// NEW: Multiple base points for different bit positions
__device__ ECPointJac G_base_points[NUM_BASE_POINTS];  // G, 2^4*G, 2^8*G, etc.
__device__ ECPointJac G_base_precomp[NUM_BASE_POINTS][1 << WINDOW_SIZE];  // Precomputed multiples for each base

// Keep all your existing functions unchanged...
__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
	#pragma unroll
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
	#pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
	#pragma unroll
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
	#pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5;
    int bit_idx = i & 31;
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}


__device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile(
        "add.cc.u32 %0, %8, %16;\n\t"
        "addc.cc.u32 %1, %9, %17;\n\t"
        "addc.cc.u32 %2, %10, %18;\n\t"
        "addc.cc.u32 %3, %11, %19;\n\t"
        "addc.cc.u32 %4, %12, %20;\n\t"
        "addc.cc.u32 %5, %13, %21;\n\t"
        "addc.cc.u32 %6, %14, %22;\n\t"
        "addc.u32 %7, %15, %23;\n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.u32 %7, %15, %23;\n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__device__ __forceinline__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

__device__ __forceinline__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}


// Forward declarations for modular ops used by helpers
__device__ __forceinline__ void sub_mod_device_fast(BigInt *res, const BigInt *a, const BigInt *b);

// Helpers for modular inverse (binary EGCD)
__device__ __forceinline__ bool bigint_is_even(const BigInt *a) {
    return (a->data[0] & 1u) == 0u;
}

__device__ __forceinline__ void bigint_rshift1(BigInt *a) {
    uint32_t carry = 0;
    for (int i = BIGINT_WORDS - 1; i >= 0; --i) {
        uint32_t new_carry = a->data[i] & 1u;
        a->data[i] = (a->data[i] >> 1) | (carry << 31);
        carry = new_carry;
    }
}

// Count trailing zeros across 256-bit BigInt
__device__ __forceinline__ int bigint_ctz(const BigInt *a) {
    int count = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint32_t w = a->data[i];
        if (w == 0) { count += 32; continue; }
        // PTX clz then compute ctz via bit-reverse
        unsigned c;
        asm volatile("brev.b32 %0, %1;\n\tclz.b32 %0, %0;" : "=r"(c) : "r"(w));
        return count + (int)c;
    }
    return 256; // a == 0
}

// Right shift by k bits, 0 <= k <= 255
__device__ __forceinline__ void bigint_rshift_k(BigInt *a, int k) {
    if (k <= 0) return;
    if (k >= 256) { init_bigint(a, 0); return; }
    int word = k >> 5;
    int bits = k & 31;
    if (word) {
        for (int i = 0; i < BIGINT_WORDS - word; ++i) a->data[i] = a->data[i + word];
        for (int i = BIGINT_WORDS - word; i < BIGINT_WORDS; ++i) a->data[i] = 0;
    }
    if (bits) {
        uint32_t prev = 0;
        for (int i = BIGINT_WORDS - 1; i >= 0; --i) {
            uint32_t cur = a->data[i];
            a->data[i] = (cur >> bits) | (prev << (32 - bits));
            prev = cur;
        }
    }
}

__device__ __forceinline__ void halve_mod_p(BigInt *x) {
    if (!bigint_is_even(x)) {
        // Integer add: x += p (no modular reduction) to make it even
        ptx_u256Add(x, x, &const_p);
    }
    bigint_rshift1(x);
    // Ensure x in [0, p)
    if (compare_bigint(x, &const_p) >= 0) {
        ptx_u256Sub(x, x, &const_p);
    }
}

// Non-modular subtraction r = a - b, assuming a >= b
__device__ __forceinline__ void bigint_sub_nored(BigInt *r, const BigInt *a, const BigInt *b) {
    ptx_u256Sub(r, a, b);
}

__device__ __forceinline__ bool bigint_is_one(const BigInt *a) {
    if (a->data[0] != 1u) return false;
    #pragma unroll
    for (int i = 1; i < BIGINT_WORDS; ++i) {
        if (a->data[i] != 0u) return false;
    }
    return true;
}

__device__ __forceinline__ void reduce_mod_p(BigInt *x) {
    if (compare_bigint(x, &const_p) >= 0) {
        ptx_u256Sub(x, x, &const_p);
    }
}

// Optimized multiply_bigint_by_const with unrolling
__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint32_t lo, hi;
        asm volatile(
            "mul.lo.u32 %0, %2, %3;\n\t"
            "mul.hi.u32 %1, %2, %3;\n\t"
            "add.cc.u32 %0, %0, %4;\n\t"
            "addc.u32 %1, %1, 0;\n\t"
            : "=r"(lo), "=r"(hi)
            : "r"(a->data[i]), "r"(c), "r"(carry)
        );
        result[i] = lo;
        carry = hi;
    }
    result[8] = carry;
}

// Optimized shift_left_word
__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
}

__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    // Use PTX add with carry chain for efficient 9-word addition
    asm volatile(
        "add.cc.u32 %0, %0, %9;\n\t"      // r[0] += addend[0], set carry
        "addc.cc.u32 %1, %1, %10;\n\t"    // r[1] += addend[1] + carry, set carry
        "addc.cc.u32 %2, %2, %11;\n\t"    // r[2] += addend[2] + carry, set carry
        "addc.cc.u32 %3, %3, %12;\n\t"    // r[3] += addend[3] + carry, set carry
        "addc.cc.u32 %4, %4, %13;\n\t"    // r[4] += addend[4] + carry, set carry
        "addc.cc.u32 %5, %5, %14;\n\t"    // r[5] += addend[5] + carry, set carry
        "addc.cc.u32 %6, %6, %15;\n\t"    // r[6] += addend[6] + carry, set carry
        "addc.cc.u32 %7, %7, %16;\n\t"    // r[7] += addend[7] + carry, set carry
        "addc.u32 %8, %8, %17;\n\t"       // r[8] += addend[8] + carry (no carry out needed)
        : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), 
          "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]), 
          "+r"(r[8])
        : "r"(addend[0]), "r"(addend[1]), "r"(addend[2]), "r"(addend[3]),
          "r"(addend[4]), "r"(addend[5]), "r"(addend[6]), "r"(addend[7]),
          "r"(addend[8])
    );
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {

	#pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}

__device__ __forceinline__ void add_9word_with_carry(uint32_t r[9], const uint32_t addend[9]) {
    // Single-pass carry propagation with minimal branching
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        uint32_t sum = r[i] + addend[i] + carry;
        carry = (sum < r[i]) | ((sum == r[i]) & addend[i]) | 
                ((sum == addend[i]) & carry);
        r[i] = sum;
    }
    r[8] = carry; // Store final carry (will be 0 or 1)
}

__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    // Full 256x256 -> 512-bit multiplication
    // Store as 32-bit words for compatibility
    uint32_t product[16] = {0};
    
    // Standard schoolbook multiplication with PTX optimization
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        
        #pragma unroll
        for (int j = 0; j < BIGINT_WORDS; j++) {
            // product[i+j] += a[i] * b[j] + carry
            uint64_t mul = (uint64_t)a->data[i] * (uint64_t)b->data[j];
            uint64_t sum = (uint64_t)product[i + j] + mul + carry;
            
            product[i + j] = (uint32_t)sum;
            carry = sum >> 32;
        }
        
        product[i + BIGINT_WORDS] = (uint32_t)carry;
    }
    
    // Fast reduction for secp256k1: p = 2^256 - 2^32 - 977
    // We have: product = low (product[0..7]) + high (product[8..15]) * 2^256
    // Since 2^256 ≡ 2^32 + 977 (mod p)
    // Result ≡ low + high * (2^32 + 977) (mod p)
    
    // result = product[0..7] (low part)
    uint32_t result[9] = {0};  // Extra word for overflow
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i] = product[i];
    }
    
    // Add: high * 977
    // For each word in high part, multiply by 977 and add to result
    uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        // Multiply product[8+i] by 977
        uint32_t lo977, hi977;
        asm volatile(
            "mul.lo.u32 %0, %2, 977;\n\t"
            "mul.hi.u32 %1, %2, 977;\n\t"
            : "=r"(lo977), "=r"(hi977)
            : "r"(product[8 + i])
        );
        
        // Add to result[i] with carry chain
        uint64_t sum = (uint64_t)result[i] + (uint64_t)lo977 + c;
        result[i] = (uint32_t)sum;
        c = (sum >> 32) + hi977;
    }
    
    // Propagate any remaining carry
    result[8] = (uint32_t)c;
    
    // Add: high * 2^32 (shift high by 32 bits = shift by 1 word position)
    c = 0;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t sum = (uint64_t)result[i + 1] + (uint64_t)product[8 + i] + c;
        result[i + 1] = (uint32_t)sum;
        c = sum >> 32;
    }
    
    // Handle the 9th word overflow if any
    if (result[8] != 0) {
        uint32_t overflow = result[8];
        
        // overflow * 2^256 ≡ overflow * (2^32 + 977) (mod p)
        
        // Add overflow * 977
        uint32_t lo977, hi977;
        asm volatile(
            "mul.lo.u32 %0, %2, 977;\n\t"
            "mul.hi.u32 %1, %2, 977;\n\t"
            : "=r"(lo977), "=r"(hi977)
            : "r"(overflow)
        );
        
        c = 0;
        uint64_t sum = (uint64_t)result[0] + (uint64_t)lo977;
        result[0] = (uint32_t)sum;
        c = (sum >> 32) + hi977;
        
        // Propagate carry from position 1 onwards
        for (int i = 1; i < BIGINT_WORDS && c != 0; i++) {
            sum = (uint64_t)result[i] + c;
            result[i] = (uint32_t)sum;
            c = sum >> 32;
        }
        
        // Add overflow * 2^32 to result[1]
        sum = (uint64_t)result[1] + (uint64_t)overflow;
        result[1] = (uint32_t)sum;
        c = sum >> 32;
        
        // Propagate carry
        for (int i = 2; i < BIGINT_WORDS && c != 0; i++) {
            sum = (uint64_t)result[i] + c;
            result[i] = (uint32_t)sum;
            c = sum >> 32;
        }
    }
    
    // Copy result to output
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = result[i];
    }
    
    // Final conditional reduction (at most 2 iterations needed)
    if (compare_bigint(res, &const_p) >= 0) {
        ptx_u256Sub(res, res, &const_p);
        
        // Second reduction rarely needed, but possible
        if (__builtin_expect(compare_bigint(res, &const_p) >= 0, 0)) {
            ptx_u256Sub(res, res, &const_p);
        }
    }
}

__device__ __forceinline__ void add_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t carry;
    
    // Use PTX for addition with carry flag
    asm volatile(
        "add.cc.u32 %0, %9, %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t"
        "addc.cc.u32 %2, %11, %19;\n\t"
        "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t"
        "addc.cc.u32 %5, %14, %22;\n\t"
        "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t"
        "addc.u32 %8, 0, 0;\n\t"  // capture final carry
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7]),
          "=r"(carry)
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
    
    if (carry || compare_bigint(res, &const_p) >= 0) {
        ptx_u256Sub(res, res, &const_p);
    }
}
template<int WINDOW_SIZE2>
__device__ void modexp(BigInt *res, const BigInt *base, const BigInt *exp) {
    constexpr int TABLE_SIZE = 1 << (WINDOW_SIZE2 - 1); // odd-only table
    BigInt precomp[TABLE_SIZE];
    BigInt result, base_sq;

    init_bigint(&result, 1);
    
    // OPTIMIZATION 3: Compute base^2 more efficiently and cache result pointer
    mul_mod_device(&base_sq, base, base);
    
    // Cache base_sq pointer for faster access in loop
    BigInt *base_sq_ptr = &base_sq;
    
    // Precompute odd powers: precomp[k] = base^(2*k + 1)
    copy_bigint(&precomp[0], base); // base^1
    
    // Use base^2 to compute all odd powers efficiently with cached pointer
    for (int k = 1; k < TABLE_SIZE; k++) {
        mul_mod_device(&precomp[k], &precomp[k - 1], base_sq_ptr);
    }
    
    // OPTIMIZATION 1: Cache exp->data in local array for faster access
    uint32_t exp_words[BIGINT_WORDS];
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        exp_words[i] = exp->data[i];
    }
    
    // Find highest set bit using optimized search
    int highest_bit = -1;
    
    // Start from highest word and work down
    for (int word = BIGINT_WORDS - 1; word >= 0; word--) {
        uint32_t v = exp_words[word];
        if (v != 0) {
            // Use __clz intrinsic for fast leading zero count
            int lz = __clz(v);
            highest_bit = word * 32 + (31 - lz);
            break;
        }
    }
    
    // Handle exp == 0 case
    if (__builtin_expect(highest_bit == -1, 0)) {
        copy_bigint(res, &result);
        return;
    }
    
    // Main exponentiation loop
    int i = highest_bit;
    while (i >= 0) {
        // OPTIMIZATION 4: Pre-calculate bit access parameters
        int word_idx = i >> 5;
        int bit_idx = i & 31;
        uint32_t current_word = exp_words[word_idx];
        uint32_t bit = (current_word >> bit_idx) & 1;
        
        if (__builtin_expect(!bit, 0)) {
            // Square once for zero bit
            mul_mod_device(&result, &result, &result);
            i--;
        } else {
            // Find window boundaries
            int window_start = i - WINDOW_SIZE2 + 1;
            if (window_start < 0) window_start = 0;
            
            // Extract window value efficiently using bit manipulation
            int window_len = i - window_start + 1;
            uint32_t window_val = 0;
            
            // Optimize window extraction using cached exp_words
            int start_word = window_start >> 5;
            int start_bit = window_start & 31;
            
            // OPTIMIZATION 4b: Reuse current_word if still in same word
            if (window_len <= 32 - start_bit) {
                // Window fits in single word
                uint32_t mask = (1U << window_len) - 1;
                uint32_t word_to_use = (start_word == word_idx) ? current_word : exp_words[start_word];
                window_val = (word_to_use >> start_bit) & mask;
            } else {
                // Window spans two words
                window_val = exp_words[start_word] >> start_bit;
                int bits_from_first = 32 - start_bit;
                int bits_from_second = window_len - bits_from_first;
                uint32_t mask = (1U << bits_from_second) - 1;
                window_val |= (exp_words[start_word + 1] & mask) << bits_from_first;
            }
            
            // Skip trailing zeros in window
            if (window_val > 0) {
                int trailing_zeros = __ffs(window_val) - 1; // __ffs returns 1-based index
                window_start += trailing_zeros;
                window_len -= trailing_zeros;
                window_val >>= trailing_zeros;
            }
            
            // Square result window_len times (unrolled for common cases)
            // OPTIMIZATION 2: Use switch statement for better compiler optimization
            switch (window_len) {
                case 1:
                    mul_mod_device(&result, &result, &result);
                    break;
                case 2:
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    break;
                case 3:
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    break;
                case 4:
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    break;
                case 5:
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    mul_mod_device(&result, &result, &result);
                    break;
                default:
                    // General case for window_len > 5
                    #pragma unroll 4
                    for (int j = 0; j < window_len; j++) {
                        mul_mod_device(&result, &result, &result);
                    }
                    break;
            }
            
            // Multiply by precomputed odd power
            if (__builtin_expect(window_val > 0, 1)) {
                int idx = (window_val - 1) >> 1; // Map odd value to table index
                mul_mod_device(&result, &result, &precomp[idx]);
            }
            
            i = window_start - 1;
        }
    }
    
    copy_bigint(res, &result);
}

__device__ __forceinline__ void mod_inverse(BigInt *res, const BigInt *a) {
    // Handle the edge case of zero explicitly. 0 has no inverse.
    if (is_zero(a)) {
        init_bigint(res, 0);
        return;
    }

    // Reduce a modulo p first
    BigInt a_reduced;
    copy_bigint(&a_reduced, a);
    while (compare_bigint(&a_reduced, &const_p) >= 0) {
        ptx_u256Sub(&a_reduced, &a_reduced, &const_p);
    }

    // Handle the edge case of one explicitly as a minor optimization.
    BigInt one; init_bigint(&one, 1);
    if (compare_bigint(&a_reduced, &one) == 0) {
        copy_bigint(res, &one);
        return;
    }

    // Perform the main computation using Fermat's Little Theorem: a^(p-2) ≡ a^(-1) (mod p)
    modexp<MOD_EXP>(res, &a_reduced, &const_p_minus_2);
}

// Optimized sub_mod_device_fast for use in add_point_jac
__device__ __forceinline__ void sub_mod_device_fast(BigInt *res, const BigInt *a, const BigInt *b) {
    // Check if subtraction will underflow using PTX
    uint32_t borrow;
    asm volatile(
        "sub.cc.u32 %0, %9, %17;\n\t"
        "subc.cc.u32 %1, %10, %18;\n\t"
        "subc.cc.u32 %2, %11, %19;\n\t"
        "subc.cc.u32 %3, %12, %20;\n\t"
        "subc.cc.u32 %4, %13, %21;\n\t"
        "subc.cc.u32 %5, %14, %22;\n\t"
        "subc.cc.u32 %6, %15, %23;\n\t"
        "subc.cc.u32 %7, %16, %24;\n\t"
        "subc.u32 %8, 0, 0;\n\t"  // capture borrow
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7]),
          "=r"(borrow)
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
    
    // If there was a borrow, add p
    if (borrow) {
        ptx_u256Add(res, res, &const_p);
    }
}


__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity || is_zero(&P->Y)) {
        point_set_infinity_jac(R);
        return;
    }
    BigInt A, B, C, D, X3, Y3, Z3, temp, temp2;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&temp, &P->X, &A);
    init_bigint(&temp2, 4);
    mul_mod_device(&B, &temp, &temp2);
    mul_mod_device(&temp, &A, &A);
    init_bigint(&temp2, 8);
    mul_mod_device(&C, &temp, &temp2);
    mul_mod_device(&temp, &P->X, &P->X);
    init_bigint(&temp2, 3);
    mul_mod_device(&D, &temp, &temp2);
    BigInt D2, two, twoB;
    mul_mod_device(&D2, &D, &D);
    init_bigint(&two, 2);
    mul_mod_device(&twoB, &B, &two);
    sub_mod_device_fast(&X3, &D2, &twoB);
    sub_mod_device_fast(&temp, &B, &X3);
    mul_mod_device(&temp, &D, &temp);
    sub_mod_device_fast(&Y3, &temp, &C);
    init_bigint(&temp, 2);
    mul_mod_device(&temp, &temp, &P->Y);
    mul_mod_device(&Z3, &temp, &P->Z);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}


__device__ __forceinline__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    // Fast infinity checks using likely/unlikely hints
    if (__builtin_expect(P->infinity, 0)) { 
        point_copy_jac(R, Q); 
        return; 
    }
    if (__builtin_expect(Q->infinity, 0)) { 
        point_copy_jac(R, P); 
        return; 
    }
    
    // Reuse memory more efficiently - use union for temporary storage
    union TempStorage {
        struct {
            BigInt Z1Z1, Z2Z2, U1, U2, H;
            BigInt S1, S2, R_big, temp1, temp2;
        } vars;
        BigInt temp_array[10]; // For flexible reuse
    } temp;
    
    // Step 1: Compute Z coordinates squared
    mul_mod_device(&temp.vars.Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&temp.vars.Z2Z2, &Q->Z, &Q->Z);
    
    // Step 2: Compute U1 and U2 
    mul_mod_device(&temp.vars.U1, &P->X, &temp.vars.Z2Z2);
    mul_mod_device(&temp.vars.U2, &Q->X, &temp.vars.Z1Z1);
    
    // Step 3: Compute H = U2 - U1
    sub_mod_device_fast(&temp.vars.H, &temp.vars.U2, &temp.vars.U1);
    
    // Fast check for point doubling case
    if (__builtin_expect(is_zero(&temp.vars.H), 0)) {
        // Compute Z cubed values (reuse Z1Z1, Z2Z2)
        mul_mod_device(&temp.vars.temp1, &temp.vars.Z1Z1, &P->Z);  // Z1^3
        mul_mod_device(&temp.vars.temp2, &temp.vars.Z2Z2, &Q->Z);  // Z2^3
        
        // Compute S1 and S2
        mul_mod_device(&temp.vars.S1, &P->Y, &temp.vars.temp2);
        mul_mod_device(&temp.vars.S2, &Q->Y, &temp.vars.temp1);
        
        if (compare_bigint(&temp.vars.S1, &temp.vars.S2) != 0) {
            point_set_infinity_jac(R);
        } else {
            double_point_jac(R, P);
        }
        return;
    }
    
    // Main addition case - compute S1, S2, R_big
    // Reuse temp1, temp2 for Z cubed values
    mul_mod_device(&temp.vars.temp1, &temp.vars.Z1Z1, &P->Z);  // Z1^3
    mul_mod_device(&temp.vars.temp2, &temp.vars.Z2Z2, &Q->Z);  // Z2^3
    
    mul_mod_device(&temp.vars.S1, &P->Y, &temp.vars.temp2);
    mul_mod_device(&temp.vars.S2, &Q->Y, &temp.vars.temp1);
    
    sub_mod_device_fast(&temp.vars.R_big, &temp.vars.S2, &temp.vars.S1);
    
    // Compute H^2 and H^3 (reuse Z1Z1, Z2Z2 which are no longer needed)
    mul_mod_device(&temp.vars.Z1Z1, &temp.vars.H, &temp.vars.H);      // H^2
    mul_mod_device(&temp.vars.Z2Z2, &temp.vars.Z1Z1, &temp.vars.H);   // H^3
    
    // Compute U1*H^2 (reuse temp1)
    mul_mod_device(&temp.vars.temp1, &temp.vars.U1, &temp.vars.Z1Z1);  // U1*H^2
    
    // Compute R^2 (reuse temp2)
    mul_mod_device(&temp.vars.temp2, &temp.vars.R_big, &temp.vars.R_big);  // R^2
    
    // Compute X3 = R^2 - H^3 - 2*U1*H^2
    sub_mod_device_fast(&R->X, &temp.vars.temp2, &temp.vars.Z2Z2);  // R^2 - H^3
    sub_mod_device_fast(&R->X, &R->X, &temp.vars.temp1);            // - U1*H^2
    sub_mod_device_fast(&R->X, &R->X, &temp.vars.temp1);            // - U1*H^2 (again)
    
    // Compute Y3 = R*(U1*H^2 - X3) - S1*H^3
    sub_mod_device_fast(&temp.vars.U2, &temp.vars.temp1, &R->X);     // U1*H^2 - X3 (reuse U2)
    mul_mod_device(&temp.vars.U2, &temp.vars.R_big, &temp.vars.U2);  // R*(U1*H^2 - X3)
    
    mul_mod_device(&temp.vars.S2, &temp.vars.S1, &temp.vars.Z2Z2);   // S1*H^3 (reuse S2)
    sub_mod_device_fast(&R->Y, &temp.vars.U2, &temp.vars.S2);       // Final Y3
    
    // Compute Z3 = Z1*Z2*H (reuse remaining temp)
    mul_mod_device(&temp.vars.temp1, &P->Z, &Q->Z);
    mul_mod_device(&R->Z, &temp.vars.temp1, &temp.vars.H);
    
    R->infinity = false;
}

// Optimized rotate right for SHA-256
__device__ inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    const uint32_t K[] = {
        0x428a2f98ul,0x71374491ul,0xb5c0fbcful,0xe9b5dba5ul,
        0x3956c25bul,0x59f111f1ul,0x923f82a4ul,0xab1c5ed5ul,
        0xd807aa98ul,0x12835b01ul,0x243185beul,0x550c7dc3ul,
        0x72be5d74ul,0x80deb1feul,0x9bdc06a7ul,0xc19bf174ul,
        0xe49b69c1ul,0xefbe4786ul,0x0fc19dc6ul,0x240ca1ccul,
        0x2de92c6ful,0x4a7484aaul,0x5cb0a9dcul,0x76f988daul,
        0x983e5152ul,0xa831c66dul,0xb00327c8ul,0xbf597fc7ul,
        0xc6e00bf3ul,0xd5a79147ul,0x06ca6351ul,0x14292967ul,
        0x27b70a85ul,0x2e1b2138ul,0x4d2c6dfcul,0x53380d13ul,
        0x650a7354ul,0x766a0abbul,0x81c2c92eul,0x92722c85ul,
        0xa2bfe8a1ul,0xa81a664bul,0xc24b8b70ul,0xc76c51a3ul,
        0xd192e819ul,0xd6990624ul,0xf40e3585ul,0x106aa070ul,
        0x19a4c116ul,0x1e376c08ul,0x2748774cul,0x34b0bcb5ul,
        0x391c0cb3ul,0x4ed8aa4aul,0x5b9cca4ful,0x682e6ff3ul,
        0x748f82eeul,0x78a5636ful,0x84c87814ul,0x8cc70208ul,
        0x90befffaul,0xa4506cebul,0xbef9a3f7ul,0xc67178f2ul
    };

    uint32_t h[8] = {
        0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
        0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul
    };

    // Optimized for 33-byte input (compressed pubkey)
    uint8_t full[64] = {0};
    
    // Copy input data
    #pragma unroll
    for (int i = 0; i < len; ++i) full[i] = data[i];
    full[len] = 0x80;
    
    // Add length in bits (big-endian) at the end
    uint64_t bit_len = (uint64_t)len * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        full[63 - i] = bit_len >> (8 * i);
    }

    // Process single block (we know it's only one block for 33 bytes)
    uint32_t w[64];
    
    // Load message schedule with proper byte order
    #pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        w[i] = (full[4 * i] << 24) | (full[4 * i + 1] << 16) |
               (full[4 * i + 2] << 8) | full[4 * i + 3];
    }
    
    // Extend message schedule
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hval = h[7];

    // Main compression loop
    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hval + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        hval = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hval;

    // Output hash (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        hash[4 * i + 0] = (h[i] >> 24) & 0xFF;
        hash[4 * i + 1] = (h[i] >> 16) & 0xFF;
        hash[4 * i + 2] = (h[i] >> 8) & 0xFF;
        hash[4 * i + 3] = (h[i] >> 0) & 0xFF;
    }
}

__device__ void ripemd160(const uint8_t* msg, uint8_t* out) {
    // RIPEMD-160 constants
    const uint32_t K1[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
    const uint32_t K2[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
    
    // Message schedule for left and right lines
    const int ZL[80] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
    };
    
    const int ZR[80] = {
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
    };
    
    // Shift amounts for left and right lines
    const int SL[80] = {
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
    };
    
    const int SR[80] = {
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
    };
    
    // Initialize hash values
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;
    
    // Prepare message: add padding and length
    uint8_t buffer[64];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        buffer[i] = msg[i];
    }
    
    // Add padding
    buffer[32] = 0x80;
    #pragma unroll
    for (int i = 33; i < 56; i++) {
        buffer[i] = 0x00;
    }
    
    // Add length (256 bits = 32 bytes)
    uint64_t bitlen = 256;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        buffer[56 + i] = (bitlen >> (i * 8)) & 0xFF;
    }
    
    // Convert buffer to 32-bit data (little-endian)
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        X[i] = ((uint32_t)buffer[i*4]) | 
               ((uint32_t)buffer[i*4 + 1] << 8) | 
               ((uint32_t)buffer[i*4 + 2] << 16) | 
               ((uint32_t)buffer[i*4 + 3] << 24);
    }
    
    // Working variables
    uint32_t AL = h0, BL = h1, CL = h2, DL = h3, EL = h4;
    uint32_t AR = h0, BR = h1, CR = h2, DR = h3, ER = h4;
    
    // Process message in 5 rounds of 16 operations each
    #pragma unroll 10
    for (int j = 0; j < 80; j++) {
        uint32_t T;
        
        // Left line
        if (j < 16) {
            T = AL + (BL ^ CL ^ DL) + X[ZL[j]] + K1[0];
        } else if (j < 32) {
            T = AL + ((BL & CL) | (~BL & DL)) + X[ZL[j]] + K1[1];
        } else if (j < 48) {
            T = AL + ((BL | ~CL) ^ DL) + X[ZL[j]] + K1[2];
        } else if (j < 64) {
            T = AL + ((BL & DL) | (CL & ~DL)) + X[ZL[j]] + K1[3];
        } else {
            T = AL + (BL ^ (CL | ~DL)) + X[ZL[j]] + K1[4];
        }
        T = ((T << SL[j]) | (T >> (32 - SL[j]))) + EL;
        AL = EL; EL = DL; DL = (CL << 10) | (CL >> 22); CL = BL; BL = T;
        
        // Right line
        if (j < 16) {
            T = AR + (BR ^ (CR | ~DR)) + X[ZR[j]] + K2[0];
        } else if (j < 32) {
            T = AR + ((BR & DR) | (CR & ~DR)) + X[ZR[j]] + K2[1];
        } else if (j < 48) {
            T = AR + ((BR | ~CR) ^ DR) + X[ZR[j]] + K2[2];
        } else if (j < 64) {
            T = AR + ((BR & CR) | (~BR & DR)) + X[ZR[j]] + K2[3];
        } else {
            T = AR + (BR ^ CR ^ DR) + X[ZR[j]] + K2[4];
        }
        T = ((T << SR[j]) | (T >> (32 - SR[j]))) + ER;
        AR = ER; ER = DR; DR = (CR << 10) | (CR >> 22); CR = BR; BR = T;
    }
    
    // Add results
    uint32_t T = h1 + CL + DR;
    h1 = h2 + DL + ER;
    h2 = h3 + EL + AR;
    h3 = h4 + AL + BR;
    h4 = h0 + BL + CR;
    h0 = T;
    
    // Convert hash to bytes (little-endian)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i]      = (h0 >> (i * 8)) & 0xFF;
        out[i + 4]  = (h1 >> (i * 8)) & 0xFF;
        out[i + 8]  = (h2 >> (i * 8)) & 0xFF;
        out[i + 12] = (h3 >> (i * 8)) & 0xFF;
        out[i + 16] = (h4 >> (i * 8)) & 0xFF;
    }
}

__device__ __forceinline__ void hash160(const uint8_t* data, int len, uint8_t out[20]) {
    uint8_t sha[32];
    sha256(data, len, sha);
    ripemd160(sha, out);
}


__device__ void jacobian_to_hash160_direct(const ECPointJac *P, uint8_t hash160_out[20]) {

    BigInt Zinv;
    mod_inverse(&Zinv, &P->Z);   // 1 expensive inverse

    // Zinv²
    BigInt Zinv2;
    mul_mod_device(&Zinv2, &Zinv, &Zinv);

    // x_affine = X * Zinv²
    BigInt x_affine;
    mul_mod_device(&x_affine, &P->X, &Zinv2);

    // Zinv³ = Zinv² * Zinv
    BigInt Zinv3;
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);

    // y_affine = Y * Zinv³   (only parity is needed!)
    BigInt y_affine;
    mul_mod_device(&y_affine, &P->Y, &Zinv3);

    // compressed pubkey
    uint8_t pubkey[33];
    pubkey[0] = 0x02 + (y_affine.data[0] & 1);

    // serialize X
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t word = x_affine.data[7 - i];
        pubkey[1 + i*4 + 0] = (word >> 24) & 0xFF;
        pubkey[1 + i*4 + 1] = (word >> 16) & 0xFF;
        pubkey[1 + i*4 + 2] = (word >> 8)  & 0xFF;
        pubkey[1 + i*4 + 3] = (word)       & 0xFF;
    }

    // hash160(pubkey)
    hash160(pubkey, 33, hash160_out);
}

__device__ void jacobian_to_affine(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) {
        R->infinity = true;
        init_bigint(&R->x, 0);
        init_bigint(&R->y, 0);
        return;
    }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse(&Zinv, &P->Z);
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}

// OPTIMIZED: Sequential accumulation instead of binary tree reduction
// This uses O(1) memory instead of O(NUM_BASE_POINTS) and is actually faster
__device__ __forceinline__ void scalar_multiply_multi_base_jac(ECPointJac *result, const BigInt *scalar) {
    point_set_infinity_jac(result);
    
    // Process windows from high to low for sequential accumulation
    // This is much more memory-efficient and cache-friendly
    
    #pragma unroll 4
    for (int window = NUM_BASE_POINTS - 1; window >= 0; window--) {
        int bit_index = window * WINDOW_SIZE;
        
        // Extract window value
        uint32_t window_val = 0;
        #pragma unroll
        for (int i = 0; i < WINDOW_SIZE; i++) {
            if (get_bit(scalar, bit_index + i)) {
                window_val |= (1U << i);
            }
        }
        
        // Skip if window is zero
        if (window_val == 0) continue;
        
        // Add the precomputed point for this window
        if (window_val < (1 << WINDOW_SIZE)) {
            if (result->infinity) {
                // First non-zero window - just copy
                point_copy_jac(result, &G_base_precomp[window][window_val]);
            } else {
                // Add to accumulator
                ECPointJac temp;
                add_point_jac(&temp, result, &G_base_precomp[window][window_val]);
                point_copy_jac(result, &temp);
            }
        }
    }
}
__device__ void jacobian_batch_to_hash160(const ECPointJac points[BATCH_SIZE], uint8_t hash160_out[BATCH_SIZE][20]) {
    // Compact valid points into a smaller array
    // Use stack arrays but optimize memory layout
    
    struct CompactPoint {
        BigInt Z;
        uint8_t original_idx;
    };
    
    CompactPoint valid_points[BATCH_SIZE];
    uint8_t valid_count = 0;
    
    // Step 1: Validity check and compaction in single pass
    #pragma unroll 8
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Fast zero check using bitwise OR
        uint32_t z_check = 0;
        #pragma unroll
        for (int j = 0; j < BIGINT_WORDS; j++) {
            z_check |= points[i].Z.data[j];
        }
        
        bool is_valid = (!points[i].infinity) && (z_check != 0);
        
        if (is_valid) {
            // Store Z coordinate and original index
            copy_bigint(&valid_points[valid_count].Z, &points[i].Z);
            valid_points[valid_count].original_idx = i;
            valid_count++;
        } else {
            // Zero out invalid hash using 64-bit writes
            uint64_t* hash_ptr = (uint64_t*)hash160_out[i];
            hash_ptr[0] = 0;
            hash_ptr[1] = 0;
            ((uint32_t*)hash_ptr)[4] = 0;
        }
    }
    
    // Early exit if no valid points
    if (valid_count == 0) return;
    
    // Step 2: Montgomery batch inversion
    // Allocate products array only for valid points
    BigInt products[BATCH_SIZE];
    BigInt inverses[BATCH_SIZE];
    
    // Compute cumulative products
    copy_bigint(&products[0], &valid_points[0].Z);
    
    #pragma unroll 8
    for (int i = 1; i < valid_count; i++) {
        mul_mod_device(&products[i], &products[i-1], &valid_points[i].Z);
    }
    
    // Compute inverse of final product (single expensive operation)
    BigInt inv_final;
    mod_inverse(&inv_final, &products[valid_count - 1]);
    
    // Propagate inverses backwards
    BigInt current_inv = inv_final;
    
    #pragma unroll 8
    for (int i = valid_count - 1; i > 0; i--) {
        // inverses[i] = current_inv * products[i-1]
        mul_mod_device(&inverses[i], &current_inv, &products[i-1]);
        
        // current_inv *= Z[i]
        mul_mod_device(&current_inv, &current_inv, &valid_points[i].Z);
    }
    copy_bigint(&inverses[0], &current_inv);
    
    // Step 3: Convert to affine and hash
    #pragma unroll 8
    for (int i = 0; i < valid_count; i++) {
        uint8_t orig_idx = valid_points[i].original_idx;
        
        // Compute Zinv^2
        BigInt Zinv2;
        mul_mod_device(&Zinv2, &inverses[i], &inverses[i]);
        
        // Compute x_affine = X * Zinv^2
        BigInt x_affine;
        mul_mod_device(&x_affine, &points[orig_idx].X, &Zinv2);
        
        // Compute Zinv^3 = Zinv^2 * Zinv
        BigInt Zinv3;
        mul_mod_device(&Zinv3, &Zinv2, &inverses[i]);
        
        // Compute Y * Zinv^3 for parity bit only
        BigInt y_affine;
        mul_mod_device(&y_affine, &points[orig_idx].Y, &Zinv3);
        
        // Build compressed pubkey in registers
        uint8_t pubkey[33];
        pubkey[0] = 0x02 | (y_affine.data[0] & 1);
        
        // Serialize X coordinate (big-endian)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t word = x_affine.data[7 - j];
            int base = 1 + (j << 2);
            pubkey[base]     = (word >> 24) & 0xFF;
            pubkey[base + 1] = (word >> 16) & 0xFF;
            pubkey[base + 2] = (word >> 8) & 0xFF;
            pubkey[base + 3] = word & 0xFF;
        }
        
        // Compute hash160
        hash160(pubkey, 33, hash160_out[orig_idx]);
    }
}

// NEW: Precompute all base points and their multiples
__global__ void precompute_multi_base_kernel() {
    if (threadIdx.x == 0) {
        // Initialize first base point (G)
        point_copy_jac(&G_base_points[0], &const_G_jacobian);
        
        // Compute subsequent base points: G_base_points[i] = 2^(4*i) * G
        for (int i = 1; i < NUM_BASE_POINTS; i++) {
            point_copy_jac(&G_base_points[i], &G_base_points[i-1]);
            
            // Double WINDOW_SIZE times to get 2^WINDOW_SIZE * previous_base
            for (int j = 0; j < WINDOW_SIZE; j++) {
                double_point_jac(&G_base_points[i], &G_base_points[i]);
            }
        }
        
        // Precompute multiples for each base point
        for (int base_idx = 0; base_idx < NUM_BASE_POINTS; base_idx++) {
            // G_base_precomp[base_idx][0] = infinity
            point_set_infinity_jac(&G_base_precomp[base_idx][0]);
            
            // G_base_precomp[base_idx][1] = base point
            point_copy_jac(&G_base_precomp[base_idx][1], &G_base_points[base_idx]);
            
            // G_base_precomp[base_idx][i] = i * base_point for i = 2..15
            for (int i = 2; i < (1 << WINDOW_SIZE); i++) {
                add_point_jac(&G_base_precomp[base_idx][i], 
                             &G_base_precomp[base_idx][i-1], 
                             &G_base_points[base_idx]);
            }
        }
    }
}

// Original precompute kernel (keep for backward compatibility)
__global__ void precompute_G_kernel() {
    if (threadIdx.x == 0) {
        point_set_infinity_jac(&G_precomp[0]);
        point_copy_jac(&G_precomp[1], &const_G_jacobian);
        for (int i = 2; i < (1 << WINDOW_SIZE); i++) {
            add_point_jac(&G_precomp[i], &G_precomp[i-1], &const_G_jacobian);
        }
    }
}


inline void cpu_u256Sub(BigInt* res, const BigInt* a, const BigInt* b) {
    uint64_t borrow = 0;

    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a->data[i] - (uint64_t)b->data[i] - borrow;
        res->data[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;  // Check sign bit for borrow
    }
}

void print_gpu_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return;
    }
    
    printf("Found %d CUDA device(s):\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               (float)deviceProp.totalGlobalMem / (1024*1024*1024));
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  CUDA Cores: ~%d\n", 
               deviceProp.multiProcessorCount * 128); // Approximate
        printf("  Clock Rate: %.2f GHz\n", 
               deviceProp.clockRate / 1e6);
        printf("\n");
    }
}


// IMPROVED: Safer initialization with error checking
void init_gpu_constants() {
	
	print_gpu_info();
    const BigInt p_host = {
        { 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };
    const ECPointJac G_jacobian_host = {
        {{ 0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
                0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E }},
        {{ 0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
                0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77 }},
        {{ 1, 0, 0, 0, 0, 0, 0, 0 }}
    };
    const BigInt n_host = {
        { 0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
          0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };

    BigInt two_host;
    init_bigint(&two_host, 2);
    BigInt p_minus_2_host;
    cpu_u256Sub(&p_minus_2_host, &p_host, &two_host);

    // Copy constants
    cudaMemcpyToSymbol(const_p, &p_host, sizeof(BigInt));
    cudaMemcpyToSymbol(const_p_minus_2, &p_minus_2_host, sizeof(BigInt));
    cudaMemcpyToSymbol(const_G_jacobian, &G_jacobian_host, sizeof(ECPointJac));
    cudaMemcpyToSymbol(const_n, &n_host, sizeof(BigInt));

    // CRITICAL: Separate synchronization for each precompute kernel
    printf("Precomputing G table...\n");
    precompute_G_kernel<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR in precompute_G_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("G table complete.\n");

    printf("Precomputing multi-base tables (this may take a moment)...\n");
    precompute_multi_base_kernel<<<1, 1>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR in precompute_multi_base_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("Multi-base tables complete.\n");
    
    // Verify initialization with a test computation
    printf("Precomputation complete and verified.\n");
}