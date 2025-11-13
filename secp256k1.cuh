// author: https://t.me/biernus
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8
#define WINDOW_SIZE 18
#define NUM_BASE_POINTS 32
#define BATCH_SIZE 150
#define MOD_EXP 5


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


__device__ __forceinline__ int bigint_ctz(const BigInt *a) {
    int count = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint32_t w = a->data[i];
        if (w == 0) { count += 32; continue; }
        
        unsigned c;
        asm volatile("brev.b32 %0, %1;\n\tclz.b32 %0, %0;" : "=r"(c) : "r"(w));
        return count + (int)c;
    }
    return 256; 
}


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
        
        ptx_u256Add(x, x, &const_p);
    }
    bigint_rshift1(x);
    
    if (compare_bigint(x, &const_p) >= 0) {
        ptx_u256Sub(x, x, &const_p);
    }
}


__device__ __forceinline__ void bigint_sub_nored(BigInt *r, const BigInt *a, const BigInt *b) {
    ptx_u256Sub(r, a, b);
}

__device__ __forceinline__ bool bigint_is_one(const BigInt *a) {
    if (a->data[0] != 1u) return false;
    
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


__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint32_t carry = 0;
    
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


__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    
    
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
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
    
    
    uint32_t product[16] = {0};
    
    
    
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        
        
        for (int j = 0; j < BIGINT_WORDS; j++) {
            
            uint64_t mul = (uint64_t)a->data[i] * (uint64_t)b->data[j];
            uint64_t sum = (uint64_t)product[i + j] + mul + carry;
            
            product[i + j] = (uint32_t)sum;
            carry = sum >> 32;
        }
        
        product[i + BIGINT_WORDS] = (uint32_t)carry;
    }
    
    
    
    
    
    
    
    uint32_t result[9] = {0};  
    
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i] = product[i];
    }
    
    
    
    uint64_t c = 0;
    
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
    
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t sum = (uint64_t)result[i + 1] + (uint64_t)product[8 + i] + c;
        result[i + 1] = (uint32_t)sum;
        c = sum >> 32;
    }
    
    
    if (result[8] != 0) {
        uint32_t overflow = result[8];
        
        
        
        
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
        
        
        for (int i = 1; i < BIGINT_WORDS && c != 0; i++) {
            sum = (uint64_t)result[i] + c;
            result[i] = (uint32_t)sum;
            c = sum >> 32;
        }
        
        
        sum = (uint64_t)result[1] + (uint64_t)overflow;
        result[1] = (uint32_t)sum;
        c = sum >> 32;
        
        
        for (int i = 2; i < BIGINT_WORDS && c != 0; i++) {
            sum = (uint64_t)result[i] + c;
            result[i] = (uint32_t)sum;
            c = sum >> 32;
        }
    }
    
    
    
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = result[i];
    }
    
    
    if (compare_bigint(res, &const_p) >= 0) {
        ptx_u256Sub(res, res, &const_p);
        
        
        if (__builtin_expect(compare_bigint(res, &const_p) >= 0, 0)) {
            ptx_u256Sub(res, res, &const_p);
        }
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
            BigInt S1, S2, R_big, temp1, temp2;
        } vars;
        BigInt temp_array[10]; 
    } temp;
    
    
    mul_mod_device(&temp.vars.Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&temp.vars.Z2Z2, &Q->Z, &Q->Z);
    
    
    mul_mod_device(&temp.vars.U1, &P->X, &temp.vars.Z2Z2);
    mul_mod_device(&temp.vars.U2, &Q->X, &temp.vars.Z1Z1);
    
    
    sub_mod_device_fast(&temp.vars.H, &temp.vars.U2, &temp.vars.U1);
    
    
    if (__builtin_expect(is_zero(&temp.vars.H), 0)) {
        
        mul_mod_device(&temp.vars.temp1, &temp.vars.Z1Z1, &P->Z);  
        mul_mod_device(&temp.vars.temp2, &temp.vars.Z2Z2, &Q->Z);  
        
        
        mul_mod_device(&temp.vars.S1, &P->Y, &temp.vars.temp2);
        mul_mod_device(&temp.vars.S2, &Q->Y, &temp.vars.temp1);
        
        if (compare_bigint(&temp.vars.S1, &temp.vars.S2) != 0) {
            point_set_infinity_jac(R);
        } else {
            double_point_jac(R, P);
        }
        return;
    }
    
    
    
    mul_mod_device(&temp.vars.temp1, &temp.vars.Z1Z1, &P->Z);  
    mul_mod_device(&temp.vars.temp2, &temp.vars.Z2Z2, &Q->Z);  
    
    mul_mod_device(&temp.vars.S1, &P->Y, &temp.vars.temp2);
    mul_mod_device(&temp.vars.S2, &Q->Y, &temp.vars.temp1);
    
    sub_mod_device_fast(&temp.vars.R_big, &temp.vars.S2, &temp.vars.S1);
    
    
    mul_mod_device(&temp.vars.Z1Z1, &temp.vars.H, &temp.vars.H);      
    mul_mod_device(&temp.vars.Z2Z2, &temp.vars.Z1Z1, &temp.vars.H);   
    
    
    mul_mod_device(&temp.vars.temp1, &temp.vars.U1, &temp.vars.Z1Z1);  
    
    
    mul_mod_device(&temp.vars.temp2, &temp.vars.R_big, &temp.vars.R_big);  
    
    
    sub_mod_device_fast(&R->X, &temp.vars.temp2, &temp.vars.Z2Z2);  
    sub_mod_device_fast(&R->X, &R->X, &temp.vars.temp1);            
    sub_mod_device_fast(&R->X, &R->X, &temp.vars.temp1);            
    
    
    sub_mod_device_fast(&temp.vars.U2, &temp.vars.temp1, &R->X);     
    mul_mod_device(&temp.vars.U2, &temp.vars.R_big, &temp.vars.U2);  
    
    mul_mod_device(&temp.vars.S2, &temp.vars.S1, &temp.vars.Z2Z2);   
    sub_mod_device_fast(&R->Y, &temp.vars.U2, &temp.vars.S2);       
    
    
    mul_mod_device(&temp.vars.temp1, &P->Z, &Q->Z);
    mul_mod_device(&R->Z, &temp.vars.temp1, &temp.vars.H);
    
    R->infinity = false;
}

// Optimized SHA-256 implementation for CUDA

// Use constant memory for K constants (better caching)
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

// Inline rotate right
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// Inline SHA-256 functions
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
    // Initialize hash values
    uint32_t h0 = 0x6a09e667ul;
    uint32_t h1 = 0xbb67ae85ul;
    uint32_t h2 = 0x3c6ef372ul;
    uint32_t h3 = 0xa54ff53aul;
    uint32_t h4 = 0x510e527ful;
    uint32_t h5 = 0x9b05688cul;
    uint32_t h6 = 0x1f83d9abul;
    uint32_t h7 = 0x5be0cd19ul;
    
    // Message schedule array
    uint32_t w[64];
    
    // Prepare message block - load data directly into w[0..15]
    
    for (int i = 0; i < 16; ++i) {
        if (i * 4 < len) {
            // Load available bytes
            uint32_t val = 0;
            
            for (int j = 0; j < 4; ++j) {
                int idx = i * 4 + j;
                if (idx < len) {
                    val |= ((uint32_t)data[idx]) << (24 - j * 8);
                } else if (idx == len) {
                    val |= 0x80u << (24 - j * 8);  // Append bit '1'
                }
            }
            w[i] = val;
        } else if (i * 4 == len) {
            w[i] = 0x80000000u;  // Padding starts here
        } else if (i == 14) {
            w[i] = 0;  // Upper 32 bits of length (always 0 for len < 512MB)
        } else if (i == 15) {
            w[i] = (uint32_t)(len * 8);  // Lower 32 bits of bit length
        } else {
            w[i] = 0;
        }
    }
    
    // Extend the first 16 words into the remaining 48 words
    
    for (int i = 16; i < 64; ++i) {
        w[i] = sigma1(w[i - 2]) + w[i - 7] + sigma0(w[i - 15]) + w[i - 16];
    }
    
    // Initialize working variables
    uint32_t a = h0, b = h1, c = h2, d = h3;
    uint32_t e = h4, f = h5, g = h6, h = h7;
    
    // Main compression loop - fully unrolled
    
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + c_K[i] + w[i];
        uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }
    
    // Add compressed chunk to current hash value
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
    h5 += f;
    h6 += g;
    h7 += h;
    
    // Produce final hash (big-endian)
    
    for (int i = 0; i < 4; ++i) {
        hash[i]      = (h0 >> (24 - i * 8)) & 0xFF;
        hash[i + 4]  = (h1 >> (24 - i * 8)) & 0xFF;
        hash[i + 8]  = (h2 >> (24 - i * 8)) & 0xFF;
        hash[i + 12] = (h3 >> (24 - i * 8)) & 0xFF;
        hash[i + 16] = (h4 >> (24 - i * 8)) & 0xFF;
        hash[i + 20] = (h5 >> (24 - i * 8)) & 0xFF;
        hash[i + 24] = (h6 >> (24 - i * 8)) & 0xFF;
        hash[i + 28] = (h7 >> (24 - i * 8)) & 0xFF;
    }
}

// Optimized RIPEMD-160 implementation for CUDA

// Use constant memory for lookup tables (cached and broadcast efficiently)
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

// Inline device functions for F operations
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

// Macro to reduce code repetition
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

__device__ void ripemd160(const uint8_t* msg, uint8_t* out) {
    // Initialize hash values
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;
    
    // Prepare message block (32 bytes + padding)
    uint32_t X[16];
    
    // Load message with coalesced memory access pattern
    
    for (int i = 0; i < 8; i++) {
        X[i] = ((uint32_t)msg[i*4]) | 
               ((uint32_t)msg[i*4 + 1] << 8) | 
               ((uint32_t)msg[i*4 + 2] << 16) | 
               ((uint32_t)msg[i*4 + 3] << 24);
    }
    
    // Padding
    X[8] = 0x00000080;  // Append bit '1' and zeros
    X[9] = 0;
    X[10] = 0;
    X[11] = 0;
    X[12] = 0;
    X[13] = 0;
    X[14] = 256;  // Length in bits (32 bytes = 256 bits)
    X[15] = 0;
    
    // Working variables
    uint32_t AL = h0, BL = h1, CL = h2, DL = h3, EL = h4;
    uint32_t AR = h0, BR = h1, CR = h2, DR = h3, ER = h4;
    
    // Main rounds - fully unrolled to eliminate branching
    // Left rounds (0-15) - F function
    
    for (int j = 0; j < 16; j++) {
        ROUND(AL, BL, CL, DL, EL, F, X[c_ZL[j]], c_SL[j], c_K1[0]);
    }
    
    // Left rounds (16-31) - G function
    
    for (int j = 16; j < 32; j++) {
        ROUND(AL, BL, CL, DL, EL, G, X[c_ZL[j]], c_SL[j], c_K1[1]);
    }
    
    // Left rounds (32-47) - H function
    
    for (int j = 32; j < 48; j++) {
        ROUND(AL, BL, CL, DL, EL, H, X[c_ZL[j]], c_SL[j], c_K1[2]);
    }
    
    // Left rounds (48-63) - I function
    
    for (int j = 48; j < 64; j++) {
        ROUND(AL, BL, CL, DL, EL, I, X[c_ZL[j]], c_SL[j], c_K1[3]);
    }
    
    // Left rounds (64-79) - J function
    
    for (int j = 64; j < 80; j++) {
        ROUND(AL, BL, CL, DL, EL, J, X[c_ZL[j]], c_SL[j], c_K1[4]);
    }
    
    // Right rounds (0-15) - J function (reversed)
    
    for (int j = 0; j < 16; j++) {
        ROUND(AR, BR, CR, DR, ER, J, X[c_ZR[j]], c_SR[j], c_K2[0]);
    }
    
    // Right rounds (16-31) - I function (reversed)
    
    for (int j = 16; j < 32; j++) {
        ROUND(AR, BR, CR, DR, ER, I, X[c_ZR[j]], c_SR[j], c_K2[1]);
    }
    
    // Right rounds (32-47) - H function (reversed)
    
    for (int j = 32; j < 48; j++) {
        ROUND(AR, BR, CR, DR, ER, H, X[c_ZR[j]], c_SR[j], c_K2[2]);
    }
    
    // Right rounds (48-63) - G function (reversed)
    
    for (int j = 48; j < 64; j++) {
        ROUND(AR, BR, CR, DR, ER, G, X[c_ZR[j]], c_SR[j], c_K2[3]);
    }
    
    // Right rounds (64-79) - F function (reversed)
    
    for (int j = 64; j < 80; j++) {
        ROUND(AR, BR, CR, DR, ER, F, X[c_ZR[j]], c_SR[j], c_K2[4]);
    }
    
    // Finalization
    uint32_t T = h1 + CL + DR;
    h1 = h2 + DL + ER;
    h2 = h3 + EL + AR;
    h3 = h4 + AL + BR;
    h4 = h0 + BL + CR;
    h0 = T;
    
    // Write output - preserve little-endian byte order
    
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

    
    // Do:
    uint8_t full_hash[20];
    hash160(pubkey, 33, full_hash);
    
    // Copy only first 10 bytes
    
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
    point_set_infinity_jac(result);
    
    
    
    
    
    for (int window = NUM_BASE_POINTS - 1; window >= 0; window--) {
        int bit_index = window * WINDOW_SIZE;
        
        
        uint32_t window_val = 0;
        
        for (int i = 0; i < WINDOW_SIZE; i++) {
            if (get_bit(scalar, bit_index + i)) {
                window_val |= (1U << i);
            }
        }
        
        
        if (window_val == 0) continue;
        
        
        if (window_val < (1 << WINDOW_SIZE)) {
            if (result->infinity) {
                
                point_copy_jac(result, &G_base_precomp[window][window_val]);
            } else {
                
                ECPointJac temp;
                add_point_jac(&temp, result, &G_base_precomp[window][window_val]);
                point_copy_jac(result, &temp);
            }
        }
    }
}

__device__ void jacobian_batch_to_hash160(const ECPointJac points[BATCH_SIZE], uint8_t hash160_out[BATCH_SIZE][20]) {
    
    
    
    struct CompactPoint {
        BigInt Z;
        uint8_t original_idx;
    };
    
    CompactPoint valid_points[BATCH_SIZE];
    uint8_t valid_count = 0;
    
    
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        
        uint32_t z_check = 0;
        
        for (int j = 0; j < BIGINT_WORDS; j++) {
            z_check |= points[i].Z.data[j];
        }
        
        bool is_valid = (!points[i].infinity) && (z_check != 0);
        
        if (is_valid) {
            
            copy_bigint(&valid_points[valid_count].Z, &points[i].Z);
            valid_points[valid_count].original_idx = i;
            valid_count++;
        } else {
            
            uint64_t* hash_ptr = (uint64_t*)hash160_out[i];
            hash_ptr[0] = 0;
            hash_ptr[1] = 0;
            ((uint32_t*)hash_ptr)[4] = 0;
        }
    }
    
    
    if (valid_count == 0) return;
    
    
    
    BigInt products[BATCH_SIZE];
    BigInt inverses[BATCH_SIZE];
    
    
    copy_bigint(&products[0], &valid_points[0].Z);
    
    
    for (int i = 1; i < valid_count; i++) {
        mul_mod_device(&products[i], &products[i-1], &valid_points[i].Z);
    }
    
    
    BigInt inv_final;
    mod_inverse(&inv_final, &products[valid_count - 1]);
    
    
    BigInt current_inv = inv_final;
    
    
    for (int i = valid_count - 1; i > 0; i--) {
        
        mul_mod_device(&inverses[i], &current_inv, &products[i-1]);
        
        
        mul_mod_device(&current_inv, &current_inv, &valid_points[i].Z);
    }
    copy_bigint(&inverses[0], &current_inv);
    
    
    
    for (int i = 0; i < valid_count; i++) {
        uint8_t orig_idx = valid_points[i].original_idx;
        
        
        BigInt Zinv2;
        mul_mod_device(&Zinv2, &inverses[i], &inverses[i]);
        
        
        BigInt x_affine;
        mul_mod_device(&x_affine, &points[orig_idx].X, &Zinv2);
        
        
        BigInt Zinv3;
        mul_mod_device(&Zinv3, &Zinv2, &inverses[i]);
        
        
        BigInt y_affine;
        mul_mod_device(&y_affine, &points[orig_idx].Y, &Zinv3);
        
        
        uint8_t pubkey[33];
        pubkey[0] = 0x02 | (y_affine.data[0] & 1);
        
        
        
        for (int j = 0; j < 8; j++) {
            uint32_t word = x_affine.data[7 - j];
            int base = 1 + (j << 2);
            pubkey[base]     = (word >> 24) & 0xFF;
            pubkey[base + 1] = (word >> 16) & 0xFF;
            pubkey[base + 2] = (word >> 8) & 0xFF;
            pubkey[base + 3] = word & 0xFF;
        }
        
        
		uint8_t hash_buffer[20];
		hash160(pubkey, 33, hash_buffer);
		memcpy(hash160_out[orig_idx], hash_buffer, 20);
    }
}

// Kernel 1: Generate base points (sequential, unavoidable)
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

// Kernel 2: Build tables in parallel (one block per base point)
__global__ void build_precomp_tables_kernel() {
    int base_idx = blockIdx.x;
    if (base_idx >= NUM_BASE_POINTS) return;
    
    if (threadIdx.x == 0) {
        point_set_infinity_jac(&G_base_precomp[base_idx][0]);
        point_copy_jac(&G_base_precomp[base_idx][1], &G_base_points[base_idx]);
        
        // Sequential within each block - dependencies require this
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
    
    // Compute idx×G using double-and-add
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

// OPTIMIZED: Specialized version for adding G (where G.Z = 1) to any point P
__device__ __forceinline__ void add_G_to_point_jac(ECPointJac *R, const ECPointJac *P) {
    // Since G.Z = 1, many operations simplify:
    // Z2Z2 = 1, Z2^3 = 1
    // U2 = G.X * Z1^2
    // S2 = G.Y * Z1^3
    
    if (__builtin_expect(P->infinity, 0)) { 
        point_copy_jac(R, &const_G_jacobian); 
        return; 
    }
    
    BigInt Z1Z1, Z1Z1Z1, U1, U2, H, S1, S2, R_big;
    BigInt H2, H3, U1H2, R2, temp;
    
    // Step 1: Z1^2 (only need to compute P's Z squared)
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    
    // Step 2: U1 = P.X, U2 = G.X * Z1^2
    copy_bigint(&U1, &P->X);
    mul_mod_device(&U2, &const_G_jacobian.X, &Z1Z1);
    
    // Step 3: H = U2 - U1
    sub_mod_device_fast(&H, &U2, &U1);
    
    // Fast check for point doubling (extremely rare with random points)
    if (__builtin_expect(is_zero(&H), 0)) {
        // Z1^3 for S comparison
        mul_mod_device(&Z1Z1Z1, &Z1Z1, &P->Z);
        
        // S1 = P.Y, S2 = G.Y * Z1^3
        copy_bigint(&S1, &P->Y);
        mul_mod_device(&S2, &const_G_jacobian.Y, &Z1Z1Z1);
        
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
        } else {
            double_point_jac(R, P);
        }
        return;
    }
    
    // Main addition case
    // Z1^3 = Z1^2 * Z1
    mul_mod_device(&Z1Z1Z1, &Z1Z1, &P->Z);
    
    // S1 = P.Y, S2 = G.Y * Z1^3
    copy_bigint(&S1, &P->Y);
    mul_mod_device(&S2, &const_G_jacobian.Y, &Z1Z1Z1);
    
    // R = S2 - S1
    sub_mod_device_fast(&R_big, &S2, &S1);
    
    // H^2 and H^3
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    
    // U1*H^2
    mul_mod_device(&U1H2, &U1, &H2);
    
    // R^2
    mul_mod_device(&R2, &R_big, &R_big);
    
    // X3 = R^2 - H^3 - 2*U1*H^2
    sub_mod_device_fast(&R->X, &R2, &H3);
    sub_mod_device_fast(&R->X, &R->X, &U1H2);
    sub_mod_device_fast(&R->X, &R->X, &U1H2);
    
    // Y3 = R*(U1*H^2 - X3) - S1*H^3
    sub_mod_device_fast(&temp, &U1H2, &R->X);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&S1, &S1, &H3);
    sub_mod_device_fast(&R->Y, &temp, &S1);
    
    // Z3 = Z1*H (since G.Z = 1)
    mul_mod_device(&R->Z, &P->Z, &H);
    
    R->infinity = false;
}