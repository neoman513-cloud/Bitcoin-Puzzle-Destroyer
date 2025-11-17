#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8
#define WINDOW_SIZE 16
#define NUM_BASE_POINTS 16
#define BATCH_SIZE 216
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


__device__ ECPointJac G_precomp[1 << WINDOW_SIZE];


__device__ ECPointJac G_base_points[NUM_BASE_POINTS];  
__device__ ECPointJac G_base_precomp[NUM_BASE_POINTS][1 << WINDOW_SIZE];  


__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
	
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
	
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
	
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
	
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

__device__ __forceinline__ void sub_mod_device_fast(BigInt *res, const BigInt *a, const BigInt *b);


__device__ __forceinline__ void bigint_sub_nored(BigInt *r, const BigInt *a, const BigInt *b) {
    ptx_u256Sub(r, a, b);
}

__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    
    asm volatile(
        "add.cc.u32 %0, %0, %9;\n\t"      
        "addc.cc.u32 %1, %1, %10;\n\t"    
        "addc.cc.u32 %2, %2, %11;\n\t"    
        "addc.cc.u32 %3, %3, %12;\n\t"    
        "addc.cc.u32 %4, %4, %13;\n\t"    
        "addc.cc.u32 %5, %5, %14;\n\t"    
        "addc.cc.u32 %6, %6, %15;\n\t"    
        "addc.cc.u32 %7, %7, %16;\n\t"    
        "addc.u32 %8, %8, %17;\n\t"       
        : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), 
          "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]), 
          "+r"(r[8])
        : "r"(addend[0]), "r"(addend[1]), "r"(addend[2]), "r"(addend[3]),
          "r"(addend[4]), "r"(addend[5]), "r"(addend[6]), "r"(addend[7]),
          "r"(addend[8])
    );
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {

	
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}

__device__ __forceinline__ void add_9word_with_carry(uint32_t r[9], const uint32_t addend[9]) {
    
    uint32_t carry = 0;
    
    for (int i = 0; i < 9; i++) {
        uint32_t sum = r[i] + addend[i] + carry;
        carry = (sum < r[i]) | ((sum == r[i]) & addend[i]) | 
                ((sum == addend[i]) & carry);
        r[i] = sum;
    }
    r[8] = carry; 
}

__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    
    
    uint32_t product[16];
    
    
    #define MULADD(i, j) \
        asm volatile( \
            "mad.lo.cc.u32 %0, %3, %4, %0;\n\t" \
            "madc.hi.cc.u32 %1, %3, %4, %1;\n\t" \
            "addc.u32 %2, %2, 0;" \
            : "+r"(c0), "+r"(c1), "+r"(c2) \
            : "r"(a->data[i]), "r"(b->data[j]) \
        );
    
    uint32_t c0, c1, c2;
    
    
    c0 = c1 = c2 = 0;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(c0) : "r"(a->data[0]), "r"(b->data[0]));
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(c1) : "r"(a->data[0]), "r"(b->data[0]));
    product[0] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 1);
    MULADD(1, 0);
    product[1] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 2);
    MULADD(1, 1);
    MULADD(2, 0);
    product[2] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 3);
    MULADD(1, 2);
    MULADD(2, 1);
    MULADD(3, 0);
    product[3] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 4);
    MULADD(1, 3);
    MULADD(2, 2);
    MULADD(3, 1);
    MULADD(4, 0);
    product[4] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 5);
    MULADD(1, 4);
    MULADD(2, 3);
    MULADD(3, 2);
    MULADD(4, 1);
    MULADD(5, 0);
    product[5] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 6);
    MULADD(1, 5);
    MULADD(2, 4);
    MULADD(3, 3);
    MULADD(4, 2);
    MULADD(5, 1);
    MULADD(6, 0);
    product[6] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 7);
    MULADD(1, 6);
    MULADD(2, 5);
    MULADD(3, 4);
    MULADD(4, 3);
    MULADD(5, 2);
    MULADD(6, 1);
    MULADD(7, 0);
    product[7] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(1, 7);
    MULADD(2, 6);
    MULADD(3, 5);
    MULADD(4, 4);
    MULADD(5, 3);
    MULADD(6, 2);
    MULADD(7, 1);
    product[8] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(2, 7);
    MULADD(3, 6);
    MULADD(4, 5);
    MULADD(5, 4);
    MULADD(6, 3);
    MULADD(7, 2);
    product[9] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(3, 7);
    MULADD(4, 6);
    MULADD(5, 5);
    MULADD(6, 4);
    MULADD(7, 3);
    product[10] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(4, 7);
    MULADD(5, 6);
    MULADD(6, 5);
    MULADD(7, 4);
    product[11] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(5, 7);
    MULADD(6, 6);
    MULADD(7, 5);
    product[12] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(6, 7);
    MULADD(7, 6);
    product[13] = c0;
    
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(7, 7);
    product[14] = c0;
    
    
    product[15] = c1;
    
    #undef MULADD
    
    
    uint32_t result[9];
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i] = product[i];
    }
    result[8] = 0;
    
    
    uint64_t c = 0;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint32_t lo977, hi977;
        asm volatile(
            "mul.lo.u32 %0, %2, 977;\n\t"
            "mul.hi.u32 %1, %2, 977;\n\t"
            : "=r"(lo977), "=r"(hi977)
            : "r"(product[8 + i])
        );
        
        uint64_t sum = (uint64_t)result[i] + (uint64_t)lo977 + c;
        result[i] = (uint32_t)sum;
        c = (sum >> 32) + hi977;
    }
    
    result[8] = (uint32_t)c;
    
    
    c = 0;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t sum = (uint64_t)result[i + 1] + (uint64_t)product[8 + i] + c;
        result[i + 1] = (uint32_t)sum;
        c = sum >> 32;
    }
    
    
    uint32_t overflow = result[8];
    uint32_t has_overflow = (uint32_t)(-(int32_t)(overflow != 0));
    uint32_t lo977, hi977;
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;\n\t"
        : "=r"(lo977), "=r"(hi977)
        : "r"(overflow)
    );
    lo977 &= has_overflow;
    hi977 &= has_overflow;
    uint32_t masked_overflow = overflow & has_overflow;
    uint64_t sum0 = (uint64_t)result[0] + (uint64_t)lo977;
    uint64_t sum1 = (uint64_t)result[1] + (uint64_t)masked_overflow + (sum0 >> 32) + hi977;
    result[0] = (uint32_t)sum0;
    result[1] = (uint32_t)sum1;
    uint64_t carry = sum1 >> 32;
    
    #pragma unroll
    for (int i = 2; i < BIGINT_WORDS; i++) {
        uint64_t sum = (uint64_t)result[i] + carry;
        result[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = result[i];
    }
    
    
    uint32_t tmp[8];
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.u32 %7, %15, %23;"
        : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3]),
          "=r"(tmp[4]), "=r"(tmp[5]), "=r"(tmp[6]), "=r"(tmp[7])
        : "r"(res->data[0]), "r"(res->data[1]), "r"(res->data[2]), "r"(res->data[3]),
          "r"(res->data[4]), "r"(res->data[5]), "r"(res->data[6]), "r"(res->data[7]),
          "r"(const_p.data[0]), "r"(const_p.data[1]), "r"(const_p.data[2]), "r"(const_p.data[3]),
          "r"(const_p.data[4]), "r"(const_p.data[5]), "r"(const_p.data[6]), "r"(const_p.data[7])
    );
    
    uint32_t borrow;
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
    uint32_t mask = ~borrow;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        res->data[i] = (tmp[i] & mask) | (res->data[i] & ~mask);
    }
}
__device__ __forceinline__ void add_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t carry;
    
    
    asm volatile(
        "add.cc.u32 %0, %9, %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t"
        "addc.cc.u32 %2, %11, %19;\n\t"
        "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t"
        "addc.cc.u32 %5, %14, %22;\n\t"
        "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t"
        "addc.u32 %8, 0, 0;\n\t"  
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
    constexpr int TABLE_SIZE = 1 << (WINDOW_SIZE2 - 1); 
    BigInt precomp[TABLE_SIZE];
    BigInt result, base_sq;

    init_bigint(&result, 1);
    
    
    mul_mod_device(&base_sq, base, base);
    
    
    BigInt *base_sq_ptr = &base_sq;
    
    
    copy_bigint(&precomp[0], base); 
    
    
    for (int k = 1; k < TABLE_SIZE; k++) {
        mul_mod_device(&precomp[k], &precomp[k - 1], base_sq_ptr);
    }
    
    
    uint32_t exp_words[BIGINT_WORDS];
    
    for (int i = 0; i < BIGINT_WORDS; i++) {
        exp_words[i] = exp->data[i];
    }
    
    
    int highest_bit = -1;
    
    
    for (int word = BIGINT_WORDS - 1; word >= 0; word--) {
        uint32_t v = exp_words[word];
        if (v != 0) {
            
            int lz = __clz(v);
            highest_bit = word * 32 + (31 - lz);
            break;
        }
    }
    
    
    if (__builtin_expect(highest_bit == -1, 0)) {
        copy_bigint(res, &result);
        return;
    }
    
    
    int i = highest_bit;
    while (i >= 0) {
        
        int word_idx = i >> 5;
        int bit_idx = i & 31;
        uint32_t current_word = exp_words[word_idx];
        uint32_t bit = (current_word >> bit_idx) & 1;
        
        if (__builtin_expect(!bit, 0)) {
            
            mul_mod_device(&result, &result, &result);
            i--;
        } else {
            
            int window_start = i - WINDOW_SIZE2 + 1;
            if (window_start < 0) window_start = 0;
            
            
            int window_len = i - window_start + 1;
            uint32_t window_val = 0;
            
            
            int start_word = window_start >> 5;
            int start_bit = window_start & 31;
            
            
            if (window_len <= 32 - start_bit) {
                
                uint32_t mask = (1U << window_len) - 1;
                uint32_t word_to_use = (start_word == word_idx) ? current_word : exp_words[start_word];
                window_val = (word_to_use >> start_bit) & mask;
            } else {
                
                window_val = exp_words[start_word] >> start_bit;
                int bits_from_first = 32 - start_bit;
                int bits_from_second = window_len - bits_from_first;
                uint32_t mask = (1U << bits_from_second) - 1;
                window_val |= (exp_words[start_word + 1] & mask) << bits_from_first;
            }
            
            
            if (window_val > 0) {
                int trailing_zeros = __ffs(window_val) - 1; 
                window_start += trailing_zeros;
                window_len -= trailing_zeros;
                window_val >>= trailing_zeros;
            }
            
            
            
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
                    
                    
                    for (int j = 0; j < window_len; j++) {
                        mul_mod_device(&result, &result, &result);
                    }
                    break;
            }
            
            
            if (__builtin_expect(window_val > 0, 1)) {
                int idx = (window_val - 1) >> 1; 
                mul_mod_device(&result, &result, &precomp[idx]);
            }
            
            i = window_start - 1;
        }
    }
    
    copy_bigint(res, &result);
}

__device__ __forceinline__ void mod_inverse(BigInt *res, const BigInt *a) {
    
    if (is_zero(a)) {
        init_bigint(res, 0);
        return;
    }

    
    BigInt a_reduced;
    copy_bigint(&a_reduced, a);
    while (compare_bigint(&a_reduced, &const_p) >= 0) {
        ptx_u256Sub(&a_reduced, &a_reduced, &const_p);
    }

    
    BigInt one; init_bigint(&one, 1);
    if (compare_bigint(&a_reduced, &one) == 0) {
        copy_bigint(res, &one);
        return;
    }

    
    modexp<MOD_EXP>(res, &a_reduced, &const_p_minus_2);
}


__device__ __forceinline__ void sub_mod_device_fast(BigInt *res, const BigInt *a, const BigInt *b) {
    
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
        "subc.u32 %8, 0, 0;\n\t"  
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7]),
          "=r"(borrow)
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
    
    
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
    
    if (__builtin_expect(P->infinity, 0)) { 
        point_copy_jac(R, Q); 
        return; 
    }
    if (__builtin_expect(Q->infinity, 0)) { 
        point_copy_jac(R, P); 
        return; 
    }
    
    union TempStorage {
        struct {
            BigInt Z1Z1, Z2Z2, U1, U2, H;
            BigInt S1, S2, R_big, HH, HHH;
        } vars;
        BigInt temp_array[9];
    } temp;
    
    
    mul_mod_device(&temp.vars.Z1Z1, &P->Z, &P->Z);
    
    mul_mod_device(&temp.vars.Z2Z2, &Q->Z, &Q->Z);
    
    
    mul_mod_device(&temp.vars.U1, &P->X, &temp.vars.Z2Z2);
    
    mul_mod_device(&temp.vars.U2, &Q->X, &temp.vars.Z1Z1);
    
    
    mul_mod_device(&temp.vars.S1, &temp.vars.Z2Z2, &Q->Z);
    mul_mod_device(&temp.vars.S1, &P->Y, &temp.vars.S1);
    
    
    mul_mod_device(&temp.vars.S2, &temp.vars.Z1Z1, &P->Z);
    mul_mod_device(&temp.vars.S2, &Q->Y, &temp.vars.S2);
    
    
    sub_mod_device_fast(&temp.vars.H, &temp.vars.U2, &temp.vars.U1);
    
    if (__builtin_expect(is_zero(&temp.vars.H), 0)) {
        if (compare_bigint(&temp.vars.S1, &temp.vars.S2) != 0) {
            point_set_infinity_jac(R);
        } else {
            double_point_jac(R, P);
        }
        return;
    }
    
    
    sub_mod_device_fast(&temp.vars.R_big, &temp.vars.S2, &temp.vars.S1);
    
    
    mul_mod_device(&temp.vars.HH, &temp.vars.H, &temp.vars.H);
    
    mul_mod_device(&temp.vars.HHH, &temp.vars.HH, &temp.vars.H);
    
    
    mul_mod_device(&temp.vars.U2, &temp.vars.U1, &temp.vars.HH);
    
    
    mul_mod_device(&R->X, &temp.vars.R_big, &temp.vars.R_big);
    sub_mod_device_fast(&R->X, &R->X, &temp.vars.HHH);
    sub_mod_device_fast(&R->X, &R->X, &temp.vars.U2);
    sub_mod_device_fast(&R->X, &R->X, &temp.vars.U2);
    
    
    sub_mod_device_fast(&temp.vars.U1, &temp.vars.U2, &R->X);
    mul_mod_device(&temp.vars.U1, &temp.vars.R_big, &temp.vars.U1);
    mul_mod_device(&temp.vars.S1, &temp.vars.S1, &temp.vars.HHH);
    sub_mod_device_fast(&R->Y, &temp.vars.U1, &temp.vars.S1);
    
    
    mul_mod_device(&R->Z, &P->Z, &Q->Z);
    mul_mod_device(&R->Z, &R->Z, &temp.vars.H);
    
    R->infinity = false;
}


__constant__ uint32_t c_K[64] = {
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


__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}


__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t Sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t Sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ void sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    uint32_t h0 = 0x6a09e667ul, h1 = 0xbb67ae85ul, h2 = 0x3c6ef372ul, h3 = 0xa54ff53aul;
    uint32_t h4 = 0x510e527ful, h5 = 0x9b05688cul, h6 = 0x1f83d9abul, h7 = 0x5be0cd19ul;
    
    uint32_t w[64];
    const uint32_t* data32 = (const uint32_t*)data;
    int len_aligned = len & ~3;
    
    
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int offset = i * 4;
        if (offset < len_aligned) {
            w[i] = __byte_perm(data32[i], 0, 0x0123);
        } else if (offset < len) {
            uint32_t val = 0;
            #pragma unroll
            for (int j = 0; j < 4 && offset + j < len; ++j) {
                val |= ((uint32_t)data[offset + j]) << (24 - j * 8);
            }
            if (offset + 4 > len) val |= 0x80u << (24 - (len - offset) * 8);
            w[i] = val;
        } else if (offset == len) {
            w[i] = 0x80000000u;
        } else {
            w[i] = (i == 15) ? (uint32_t)(len * 8) : 0;
        }
    }
    
    
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = sigma0(w[i - 15]);
        uint32_t s1 = sigma1(w[i - 2]);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    
    uint32_t a = h0, b = h1, c = h2, d = h3;
    uint32_t e = h4, f = h5, g = h6, h = h7;
    
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t s1 = Sigma1(e);
        uint32_t ch = Ch(e, f, g);
        uint32_t temp1 = h + s1 + ch + c_K[i] + w[i];
        uint32_t s0 = Sigma0(a);
        uint32_t maj = Maj(a, b, c);
        uint32_t temp2 = s0 + maj;
        
        h = g; g = f; f = e;
        e = d + temp1;
        d = c; c = b; b = a;
        a = temp1 + temp2;
    }
    
    h0 += a; h1 += b; h2 += c; h3 += d;
    h4 += e; h5 += f; h6 += g; h7 += h;
    
    
    uint32_t* out = (uint32_t*)hash;
    out[0] = __byte_perm(h0, 0, 0x0123);
    out[1] = __byte_perm(h1, 0, 0x0123);
    out[2] = __byte_perm(h2, 0, 0x0123);
    out[3] = __byte_perm(h3, 0, 0x0123);
    out[4] = __byte_perm(h4, 0, 0x0123);
    out[5] = __byte_perm(h5, 0, 0x0123);
    out[6] = __byte_perm(h6, 0, 0x0123);
    out[7] = __byte_perm(h7, 0, 0x0123);
}




__constant__ uint32_t c_K1[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
__constant__ uint32_t c_K2[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

__constant__ int c_ZL[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

__constant__ int c_ZR[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

__constant__ int c_SL[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

__constant__ int c_SR[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};


__device__ __forceinline__ uint32_t F(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ y ^ z;
}

__device__ __forceinline__ uint32_t G(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (~x & z);
}

__device__ __forceinline__ uint32_t H(uint32_t x, uint32_t y, uint32_t z) {
    return (x | ~y) ^ z;
}

__device__ __forceinline__ uint32_t I(uint32_t x, uint32_t y, uint32_t z) {
    return (x & z) | (y & ~z);
}

__device__ __forceinline__ uint32_t J(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ (y | ~z);
}

__device__ __forceinline__ uint32_t ROL(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}


#define ROUND(a, b, c, d, e, func, x, s, k) \
    do { \
        uint32_t t = a + func(b, c, d) + x + k; \
        t = ROL(t, s) + e; \
        a = e; \
        e = d; \
        d = ROL(c, 10); \
        c = b; \
        b = t; \
    } while(0)
__device__ void ripemd160(const uint8_t* __restrict__ msg, 
                          uint8_t* __restrict__ out) {
    
    const uint32_t* msg32 = reinterpret_cast<const uint32_t*>(msg);
    uint32_t X[16];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X[i] = msg32[i];
    }
    
    X[8] = 0x80; X[9] = 0; X[10] = 0; X[11] = 0;
    X[12] = 0; X[13] = 0; X[14] = 256; X[15] = 0;
    
    uint32_t AL = 0x67452301, BL = 0xEFCDAB89, CL = 0x98BADCFE;
    uint32_t DL = 0x10325476, EL = 0xC3D2E1F0;
    uint32_t AR = AL, BR = BL, CR = CL, DR = DL, ER = EL;
 
    for (int j = 0; j < 16; j++) {
        ROUND(AL, BL, CL, DL, EL, F, X[c_ZL[j]], c_SL[j], c_K1[0]);
    }
    
    
    
    for (int j = 16; j < 32; j++) {
        ROUND(AL, BL, CL, DL, EL, G, X[c_ZL[j]], c_SL[j], c_K1[1]);
    }
    
    
    
    for (int j = 32; j < 48; j++) {
        ROUND(AL, BL, CL, DL, EL, H, X[c_ZL[j]], c_SL[j], c_K1[2]);
    }
    
    
    
    for (int j = 48; j < 64; j++) {
        ROUND(AL, BL, CL, DL, EL, I, X[c_ZL[j]], c_SL[j], c_K1[3]);
    }
    
    
    
    for (int j = 64; j < 80; j++) {
        ROUND(AL, BL, CL, DL, EL, J, X[c_ZL[j]], c_SL[j], c_K1[4]);
    }
    
    
    
    for (int j = 0; j < 16; j++) {
        ROUND(AR, BR, CR, DR, ER, J, X[c_ZR[j]], c_SR[j], c_K2[0]);
    }
    
    
    
    for (int j = 16; j < 32; j++) {
        ROUND(AR, BR, CR, DR, ER, I, X[c_ZR[j]], c_SR[j], c_K2[1]);
    }
    
    
    
    for (int j = 32; j < 48; j++) {
        ROUND(AR, BR, CR, DR, ER, H, X[c_ZR[j]], c_SR[j], c_K2[2]);
    }
    
    
    
    for (int j = 48; j < 64; j++) {
        ROUND(AR, BR, CR, DR, ER, G, X[c_ZR[j]], c_SR[j], c_K2[3]);
    }
    
    
    
    for (int j = 64; j < 80; j++) {
        ROUND(AR, BR, CR, DR, ER, F, X[c_ZR[j]], c_SR[j], c_K2[4]);
    }
    
    
    
    uint32_t T = 0xEFCDAB89 + CL + DR;
    
    
    uint32_t* out32 = reinterpret_cast<uint32_t*>(out);
    out32[0] = T;
    out32[1] = 0x98BADCFE + DL + ER;
    out32[2] = 0x10325476 + EL + AR;
    out32[3] = 0xC3D2E1F0 + AL + BR;
    out32[4] = 0x67452301 + BL + CR;
}
__device__ __forceinline__ void hash160(const uint8_t* data, int len, uint8_t out[20]) {
    uint8_t sha[32];
    sha256(data, len, sha);
    ripemd160(sha, out);
}


__device__ void jacobian_to_hash160_direct(const ECPointJac *P, uint8_t hash160_out[20]) {

    BigInt Zinv;
    mod_inverse(&Zinv, &P->Z);   

    
    BigInt Zinv2;
    mul_mod_device(&Zinv2, &Zinv, &Zinv);

    
    BigInt x_affine;
    mul_mod_device(&x_affine, &P->X, &Zinv2);

    
    BigInt Zinv3;
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);

    
    BigInt y_affine;
    mul_mod_device(&y_affine, &P->Y, &Zinv3);

    
    uint8_t pubkey[33];
    pubkey[0] = 0x02 + (y_affine.data[0] & 1);

    
    
    for (int i = 0; i < 8; i++) {
        uint32_t word = x_affine.data[7 - i];
        pubkey[1 + i*4 + 0] = (word >> 24) & 0xFF;
        pubkey[1 + i*4 + 1] = (word >> 16) & 0xFF;
        pubkey[1 + i*4 + 2] = (word >> 8)  & 0xFF;
        pubkey[1 + i*4 + 3] = (word)       & 0xFF;
    }

    
    
    uint8_t full_hash[20];
    hash160(pubkey, 33, full_hash);
    
    
    
    for (int i = 0; i < 10; i++) {
        hash160_out[i] = full_hash[i];
    }
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

__device__ __forceinline__ void scalar_multiply_multi_base_jac(ECPointJac *result, const BigInt *scalar) {
    
    int first_window = -1;
    
    #pragma unroll
    for (int window = NUM_BASE_POINTS - 1; window >= 0; window--) {
        int bit_index = window * WINDOW_SIZE;
        uint32_t word_idx = bit_index >> 5;  
        uint32_t bit_offset = bit_index & 31; 
        
        
        uint32_t window_val = scalar->data[word_idx] >> bit_offset;
        if (bit_offset + WINDOW_SIZE > 32) {
            window_val |= scalar->data[word_idx + 1] << (32 - bit_offset);
        }
        window_val &= (1U << WINDOW_SIZE) - 1;
        
        if (window_val != 0) {
            
            *result = G_base_precomp[window][window_val];
            first_window = window;
            break;
        }
    }
    
    
    if (first_window == -1) {
        point_set_infinity_jac(result);
        return;
    }
    
    
    #pragma unroll
    for (int window = first_window - 1; window >= 0; window--) {
        int bit_index = window * WINDOW_SIZE;
        uint32_t word_idx = bit_index >> 5;
        uint32_t bit_offset = bit_index & 31;
        
        uint32_t window_val = scalar->data[word_idx] >> bit_offset;
        if (bit_offset + WINDOW_SIZE > 32) {
            window_val |= scalar->data[word_idx + 1] << (32 - bit_offset);
        }
        window_val &= (1U << WINDOW_SIZE) - 1;
        
		ECPointJac temp = G_base_precomp[window][window_val];
		if (window_val != 0) {
			add_point_jac(result, result, &temp);
		}
    }
}

__device__ void jacobian_batch_to_hash160(const ECPointJac points[BATCH_SIZE], uint8_t hash160_out[BATCH_SIZE][20]) {
    
    uint8_t valid_map[BATCH_SIZE];
    uint8_t valid_count = 0;
    
    
    #pragma unroll
    for (int i = 0; i < BATCH_SIZE; i++) {
        uint32_t z_check = points[i].Z.data[0] | points[i].Z.data[1] | 
                          points[i].Z.data[2] | points[i].Z.data[3] |
                          points[i].Z.data[4] | points[i].Z.data[5] | 
                          points[i].Z.data[6] | points[i].Z.data[7];
        
        bool is_valid = (!points[i].infinity) & (z_check != 0);
        
        if (is_valid) {
            valid_map[valid_count++] = i;
        } else {
            *((uint64_t*)hash160_out[i]) = 0;
            *((uint64_t*)(hash160_out[i] + 8)) = 0;
            *((uint32_t*)(hash160_out[i] + 16)) = 0;
        }
    }
    
    if (valid_count == 0) return;
    
    
    BigInt products[BATCH_SIZE];
    BigInt inverses[BATCH_SIZE];
    
    copy_bigint(&products[0], &points[valid_map[0]].Z);
    
    for (int i = 1; i < valid_count; i++) {
        mul_mod_device(&products[i], &products[i-1], &points[valid_map[i]].Z);
    }
    
    BigInt current_inv;
    mod_inverse(&current_inv, &products[valid_count - 1]);
    
    for (int i = valid_count - 1; i > 0; i--) {
        mul_mod_device(&inverses[i], &current_inv, &products[i-1]);
        mul_mod_device(&current_inv, &current_inv, &points[valid_map[i]].Z);
    }
    copy_bigint(&inverses[0], &current_inv);
    
    
    for (int i = 0; i < valid_count; i++) {
        uint8_t idx = valid_map[i];
        
        BigInt Zinv2, Zinv3;
        mul_mod_device(&Zinv2, &inverses[i], &inverses[i]);
        mul_mod_device(&Zinv3, &Zinv2, &inverses[i]);
        
        BigInt x_affine, y_affine;
        mul_mod_device(&x_affine, &points[idx].X, &Zinv2);
        mul_mod_device(&y_affine, &points[idx].Y, &Zinv3);
        
        
        uint8_t pubkey[33];
        pubkey[0] = 0x02 | (y_affine.data[0] & 1);
        
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t word = x_affine.data[7 - j];
            int base = 1 + (j << 2);
            pubkey[base]     = word >> 24;
            pubkey[base + 1] = word >> 16;
            pubkey[base + 2] = word >> 8;
            pubkey[base + 3] = word;
        }
        
        hash160(pubkey, 33, hash160_out[idx]);
    }
}


__global__ void generate_base_points_kernel() {
    if (threadIdx.x == 0) {
        point_copy_jac(&G_base_points[0], &const_G_jacobian);
        
        for (int i = 1; i < NUM_BASE_POINTS; i++) {
            point_copy_jac(&G_base_points[i], &G_base_points[i-1]);
            
            for (int j = 0; j < WINDOW_SIZE; j++) {
                double_point_jac(&G_base_points[i], &G_base_points[i]);
            }
        }
    }
}


__global__ void build_precomp_tables_kernel() {
    int base_idx = blockIdx.x;
    if (base_idx >= NUM_BASE_POINTS) return;
    
    if (threadIdx.x == 0) {
        point_set_infinity_jac(&G_base_precomp[base_idx][0]);
        point_copy_jac(&G_base_precomp[base_idx][1], &G_base_points[base_idx]);
        
        
        for (int i = 2; i < (1 << WINDOW_SIZE); i++) {
            add_point_jac(&G_base_precomp[base_idx][i], 
                         &G_base_precomp[base_idx][i-1], 
                         &G_base_points[base_idx]);
        }
    }
}


__global__ void precompute_G_kernel_parallel() {
    const int TABLE_SIZE = 1 << WINDOW_SIZE;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx == 0) {
        point_set_infinity_jac(&G_precomp[0]);
        return;
    }
    
    if (idx == 1) {
        point_copy_jac(&G_precomp[1], &const_G_jacobian);
        return;
    }
    
    if (idx >= TABLE_SIZE) return;
    
    
    ECPointJac result;
    point_set_infinity_jac(&result);
    
    ECPointJac base;
    point_copy_jac(&base, &const_G_jacobian);
    
    int n = idx;
    while (n > 0) {
        if (n & 1) {
            if (result.infinity) {
                point_copy_jac(&result, &base);
            } else {
                ECPointJac temp;
                add_point_jac(&temp, &result, &base);
                point_copy_jac(&result, &temp);
            }
        }
        
        if (n > 1) {
            ECPointJac temp;
            double_point_jac(&temp, &base);
            point_copy_jac(&base, &temp);
        }
        
        n >>= 1;
    }
    
    point_copy_jac(&G_precomp[idx], &result);
}


inline void cpu_u256Sub(BigInt* res, const BigInt* a, const BigInt* b) {
    uint64_t borrow = 0;

    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a->data[i] - (uint64_t)b->data[i] - borrow;
        res->data[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;  
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
               deviceProp.multiProcessorCount * 128); 
        printf("  Clock Rate: %.2f GHz\n", 
               deviceProp.clockRate / 1e6);
        printf("\n");
    }
}



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

    
    cudaMemcpyToSymbol(const_p, &p_host, sizeof(BigInt));
    cudaMemcpyToSymbol(const_p_minus_2, &p_minus_2_host, sizeof(BigInt));
    cudaMemcpyToSymbol(const_G_jacobian, &G_jacobian_host, sizeof(ECPointJac));
    cudaMemcpyToSymbol(const_n, &n_host, sizeof(BigInt));

    
    printf("Precomputing G table...\n");
	int threads = 256;
	int blocks = ((1 << WINDOW_SIZE) + threads - 1) / threads;
	precompute_G_kernel_parallel<<<blocks, threads>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR in precompute_G_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("G table complete.\n");

    printf("Precomputing multi-base tables (this may take a moment)...\n");
    generate_base_points_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    build_precomp_tables_kernel<<<NUM_BASE_POINTS, 1>>>();
    cudaDeviceSynchronize();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR in precompute_multi_base_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("Multi-base tables complete.\n");
    
    
    printf("Precomputation complete and verified.\n");
}


__device__ __forceinline__ void add_G_to_point_jac(ECPointJac *R, const ECPointJac *P) {
    
    
    
    
    
    if (__builtin_expect(P->infinity, 0)) { 
        point_copy_jac(R, &const_G_jacobian); 
        return; 
    }
    
    BigInt Z1Z1, Z1Z1Z1, U1, U2, H, S1, S2, R_big;
    BigInt H2, H3, U1H2, R2, temp;
    
    
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    
    
    copy_bigint(&U1, &P->X);
    mul_mod_device(&U2, &const_G_jacobian.X, &Z1Z1);
    
    
    sub_mod_device_fast(&H, &U2, &U1);
    
    
    if (__builtin_expect(is_zero(&H), 0)) {
        
        mul_mod_device(&Z1Z1Z1, &Z1Z1, &P->Z);
        
        
        copy_bigint(&S1, &P->Y);
        mul_mod_device(&S2, &const_G_jacobian.Y, &Z1Z1Z1);
        
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
        } else {
            double_point_jac(R, P);
        }
        return;
    }
    
    
    
    mul_mod_device(&Z1Z1Z1, &Z1Z1, &P->Z);
    
    
    copy_bigint(&S1, &P->Y);
    mul_mod_device(&S2, &const_G_jacobian.Y, &Z1Z1Z1);
    
    
    sub_mod_device_fast(&R_big, &S2, &S1);
    
    
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    
    
    mul_mod_device(&U1H2, &U1, &H2);
    
    
    mul_mod_device(&R2, &R_big, &R_big);
    
    
    sub_mod_device_fast(&R->X, &R2, &H3);
    sub_mod_device_fast(&R->X, &R->X, &U1H2);
    sub_mod_device_fast(&R->X, &R->X, &U1H2);
    
    
    sub_mod_device_fast(&temp, &U1H2, &R->X);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&S1, &S1, &H3);
    sub_mod_device_fast(&R->Y, &temp, &S1);
    
    
    mul_mod_device(&R->Z, &P->Z, &H);
    
    R->infinity = false;
}