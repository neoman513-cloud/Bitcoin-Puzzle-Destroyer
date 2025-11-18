#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8
#define WINDOW_SIZE 16
#define NUM_BASE_POINTS 16
#define BATCH_SIZE 224
#define MOD_EXP 4


struct __align__(16) BigInt {
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

__device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
    
    uint4 *dest_vec = (uint4*)dest->data;
    const uint4 *src_vec = (const uint4*)src->data;
    
    dest_vec[0] = src_vec[0];  
    dest_vec[1] = src_vec[1];
}

__device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    uint32_t gt = 0, lt = 0;
    
    for (int i = 7; i >= 0; i--) {
        uint32_t a_word = a->data[i];
        uint32_t b_word = b->data[i];
        
        
        uint32_t gt_mask = (a_word > b_word) ? 0xFFFFFFFF : 0;
        uint32_t lt_mask = (a_word < b_word) ? 0xFFFFFFFF : 0;
        
        
        uint32_t update_mask = (gt == 0 && lt == 0) ? 0xFFFFFFFF : 0;
        gt |= (gt_mask & update_mask);
        lt |= (lt_mask & update_mask);
    }
    
    return gt ? 1 : (lt ? -1 : 0);
}

__device__ __forceinline__ bool is_zero(const BigInt *a) {
    uint32_t result;
    
    asm volatile(
        "{\n\t"
        ".reg .u32 t0, t1, t2, t3;\n\t"
        "or.b32 t0, %1, %2;\n\t"
        "or.b32 t1, %3, %4;\n\t"
        "or.b32 t2, %5, %6;\n\t"
        "or.b32 t3, %7, %8;\n\t"
        "or.b32 t0, t0, t1;\n\t"
        "or.b32 t2, t2, t3;\n\t"
        "or.b32 %0, t0, t2;\n\t"
        "}"
        : "=r"(result)
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7])
    );
    
    return (result == 0);
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
    
    #define MULADD(i, j) \
        asm volatile( \
            "mad.lo.cc.u32 %0, %3, %4, %0;\n\t" \
            "madc.hi.cc.u32 %1, %3, %4, %1;\n\t" \
            "addc.u32 %2, %2, 0;" \
            : "+r"(c0), "+r"(c1), "+r"(c2) \
            : "r"(a->data[i]), "r"(b->data[j]) \
        );
    
    uint32_t c0, c1, c2;
    uint32_t result[8];
    uint32_t prod_high[8];
    
    c0 = c1 = c2 = 0;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(c0) : "r"(a->data[0]), "r"(b->data[0]));
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(c1) : "r"(a->data[0]), "r"(b->data[0]));
    result[0] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 1);
    MULADD(1, 0);
    result[1] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 2);
    MULADD(1, 1);
    MULADD(2, 0);
    result[2] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 3);
    MULADD(1, 2);
    MULADD(2, 1);
    MULADD(3, 0);
    result[3] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 4);
    MULADD(1, 3);
    MULADD(2, 2);
    MULADD(3, 1);
    MULADD(4, 0);
    result[4] = c0;

    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 5);
    MULADD(1, 4);
    MULADD(2, 3);
    MULADD(3, 2);
    MULADD(4, 1);
    MULADD(5, 0);
    result[5] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 6);
    MULADD(1, 5);
    MULADD(2, 4);
    MULADD(3, 3);
    MULADD(4, 2);
    MULADD(5, 1);
    MULADD(6, 0);
    result[6] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(0, 7);
    MULADD(1, 6);
    MULADD(2, 5);
    MULADD(3, 4);
    MULADD(4, 3);
    MULADD(5, 2);
    MULADD(6, 1);
    MULADD(7, 0);
    result[7] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(1, 7);
    MULADD(2, 6);
    MULADD(3, 5);
    MULADD(4, 4);
    MULADD(5, 3);
    MULADD(6, 2);
    MULADD(7, 1);
    prod_high[0] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(2, 7);
    MULADD(3, 6);
    MULADD(4, 5);
    MULADD(5, 4);
    MULADD(6, 3);
    MULADD(7, 2);
    prod_high[1] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(3, 7);
    MULADD(4, 6);
    MULADD(5, 5);
    MULADD(6, 4);
    MULADD(7, 3);
    prod_high[2] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(4, 7);
    MULADD(5, 6);
    MULADD(6, 5);
    MULADD(7, 4);
    prod_high[3] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(5, 7);
    MULADD(6, 6);
    MULADD(7, 5);
    prod_high[4] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(6, 7);
    MULADD(7, 6);
    prod_high[5] = c0;
    
    c0 = c1; c1 = c2; c2 = 0;
    MULADD(7, 7);
    prod_high[6] = c0;
    
    prod_high[7] = c1;
    
    #undef MULADD
    
    uint32_t lo977, hi977;
    uint64_t sum, carry;
    
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
        : "=r"(lo977), "=r"(hi977)
        : "r"(prod_high[0])
    );
    sum = (uint64_t)result[0] + (uint64_t)lo977;
    result[0] = (uint32_t)sum;
    carry = (sum >> 32) + hi977;
    
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
        : "=r"(lo977), "=r"(hi977)
        : "r"(prod_high[1])
    );
    sum = (uint64_t)result[1] + (uint64_t)lo977 + carry;
    result[1] = (uint32_t)sum;
    carry = (sum >> 32) + hi977;
    
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
        : "=r"(lo977), "=r"(hi977)
        : "r"(prod_high[2])
    );
    sum = (uint64_t)result[2] + (uint64_t)lo977 + carry;
    result[2] = (uint32_t)sum;
    carry = (sum >> 32) + hi977;
    
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
        : "=r"(lo977), "=r"(hi977)
        : "r"(prod_high[3])
    );
    sum = (uint64_t)result[3] + (uint64_t)lo977 + carry;
    result[3] = (uint32_t)sum;
    carry = (sum >> 32) + hi977;
    
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
        : "=r"(lo977), "=r"(hi977)
        : "r"(prod_high[4])
    );
    sum = (uint64_t)result[4] + (uint64_t)lo977 + carry;
    result[4] = (uint32_t)sum;
    carry = (sum >> 32) + hi977;
    
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
        : "=r"(lo977), "=r"(hi977)
        : "r"(prod_high[5])
    );
    sum = (uint64_t)result[5] + (uint64_t)lo977 + carry;
    result[5] = (uint32_t)sum;
    carry = (sum >> 32) + hi977;
    
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
        : "=r"(lo977), "=r"(hi977)
        : "r"(prod_high[6])
    );
    sum = (uint64_t)result[6] + (uint64_t)lo977 + carry;
    result[6] = (uint32_t)sum;
    carry = (sum >> 32) + hi977;
    
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
        : "=r"(lo977), "=r"(hi977)
        : "r"(prod_high[7])
    );
    sum = (uint64_t)result[7] + (uint64_t)lo977 + carry;
    result[7] = (uint32_t)sum;
    uint32_t overflow = (uint32_t)((sum >> 32) + hi977);
    
    carry = 0;

    sum = (uint64_t)result[1] + (uint64_t)prod_high[0] + carry;
    result[1] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[2] + (uint64_t)prod_high[1] + carry;
    result[2] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[3] + (uint64_t)prod_high[2] + carry;
    result[3] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[4] + (uint64_t)prod_high[3] + carry;
    result[4] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[5] + (uint64_t)prod_high[4] + carry;
    result[5] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[6] + (uint64_t)prod_high[5] + carry;
    result[6] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[7] + (uint64_t)prod_high[6] + carry;
    result[7] = (uint32_t)sum;
    carry = sum >> 32;
    
    overflow += prod_high[7] + (uint32_t)carry;
    
    uint32_t has_overflow = (uint32_t)(-(int32_t)(overflow != 0));
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;"
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
    carry = sum1 >> 32;
    
    sum = (uint64_t)result[2] + carry;
    result[2] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[3] + carry;
    result[3] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[4] + carry;
    result[4] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[5] + carry;
    result[5] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[6] + carry;
    result[6] = (uint32_t)sum;
    carry = sum >> 32;
    
    sum = (uint64_t)result[7] + carry;
    result[7] = (uint32_t)sum;
    
    
    for (int i = 0; i < 8; i++) {
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
    
    res->data[0] = (tmp[0] & mask) | (res->data[0] & ~mask);
    res->data[1] = (tmp[1] & mask) | (res->data[1] & ~mask);
    res->data[2] = (tmp[2] & mask) | (res->data[2] & ~mask);
    res->data[3] = (tmp[3] & mask) | (res->data[3] & ~mask);
    res->data[4] = (tmp[4] & mask) | (res->data[4] & ~mask);
    res->data[5] = (tmp[5] & mask) | (res->data[5] & ~mask);
    res->data[6] = (tmp[6] & mask) | (res->data[6] & ~mask);
    res->data[7] = (tmp[7] & mask) | (res->data[7] & ~mask);
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
        
        uint4 *R_x = (uint4*)R->X.data;
        uint4 *R_y = (uint4*)R->Y.data;
        uint4 *R_z = (uint4*)R->Z.data;
        const uint4 *Q_x = (const uint4*)Q->X.data;
        const uint4 *Q_y = (const uint4*)Q->Y.data;
        const uint4 *Q_z = (const uint4*)Q->Z.data;
        
        R_x[0] = Q_x[0]; R_x[1] = Q_x[1];
        R_y[0] = Q_y[0]; R_y[1] = Q_y[1];
        R_z[0] = Q_z[0]; R_z[1] = Q_z[1];
        R->infinity = Q->infinity;
        return; 
    }
    if (__builtin_expect(Q->infinity, 0)) { 
        uint4 *R_x = (uint4*)R->X.data;
        uint4 *R_y = (uint4*)R->Y.data;
        uint4 *R_z = (uint4*)R->Z.data;
        const uint4 *P_x = (const uint4*)P->X.data;
        const uint4 *P_y = (const uint4*)P->Y.data;
        const uint4 *P_z = (const uint4*)P->Z.data;
        
        R_x[0] = P_x[0]; R_x[1] = P_x[1];
        R_y[0] = P_y[0]; R_y[1] = P_y[1];
        R_z[0] = P_z[0]; R_z[1] = P_z[1];
        R->infinity = P->infinity;
        return; 
    }
    
    
    BigInt Z1Z1, Z2Z2, U1, U2, H, S1, S2, R_big;
    BigInt HH, HHH, temp;
    
    
    
    
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    
    
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    
    
    
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.cc.u32 %7, %15, %23;\n\t"
        : "=r"(H.data[0]), "=r"(H.data[1]), "=r"(H.data[2]), "=r"(H.data[3]),
          "=r"(H.data[4]), "=r"(H.data[5]), "=r"(H.data[6]), "=r"(H.data[7])
        : "r"(U2.data[0]), "r"(U2.data[1]), "r"(U2.data[2]), "r"(U2.data[3]),
          "r"(U2.data[4]), "r"(U2.data[5]), "r"(U2.data[6]), "r"(U2.data[7]),
          "r"(U1.data[0]), "r"(U1.data[1]), "r"(U1.data[2]), "r"(U1.data[3]),
          "r"(U1.data[4]), "r"(U1.data[5]), "r"(U1.data[6]), "r"(U1.data[7])
    );
    
    
    uint32_t borrow;
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
    if (borrow) {
        asm volatile(
            "add.cc.u32 %0, %0, %8;\n\t"
            "addc.cc.u32 %1, %1, %9;\n\t"
            "addc.cc.u32 %2, %2, %10;\n\t"
            "addc.cc.u32 %3, %3, %11;\n\t"
            "addc.cc.u32 %4, %4, %12;\n\t"
            "addc.cc.u32 %5, %5, %13;\n\t"
            "addc.cc.u32 %6, %6, %14;\n\t"
            "addc.u32 %7, %7, %15;\n\t"
            : "+r"(H.data[0]), "+r"(H.data[1]), "+r"(H.data[2]), "+r"(H.data[3]),
              "+r"(H.data[4]), "+r"(H.data[5]), "+r"(H.data[6]), "+r"(H.data[7])
            : "r"(const_p.data[0]), "r"(const_p.data[1]), "r"(const_p.data[2]), "r"(const_p.data[3]),
              "r"(const_p.data[4]), "r"(const_p.data[5]), "r"(const_p.data[6]), "r"(const_p.data[7])
        );
    }
    
    
    uint32_t h_check;
    asm volatile(
        "{\n\t"
        ".reg .u32 t0, t1, t2, t3;\n\t"
        "or.b32 t0, %1, %2;\n\t"
        "or.b32 t1, %3, %4;\n\t"
        "or.b32 t2, %5, %6;\n\t"
        "or.b32 t3, %7, %8;\n\t"
        "or.b32 t0, t0, t1;\n\t"
        "or.b32 t2, t2, t3;\n\t"
        "or.b32 %0, t0, t2;\n\t"
        "}"
        : "=r"(h_check)
        : "r"(H.data[0]), "r"(H.data[1]), "r"(H.data[2]), "r"(H.data[3]),
          "r"(H.data[4]), "r"(H.data[5]), "r"(H.data[6]), "r"(H.data[7])
    );
    
    
    if (__builtin_expect(h_check == 0, 0)) {
        BigInt Z1Z1Z1, Z2Z2Z2;
        
        mul_mod_device(&Z1Z1Z1, &Z1Z1, &P->Z);
        mul_mod_device(&Z2Z2Z2, &Z2Z2, &Q->Z);
        mul_mod_device(&S1, &P->Y, &Z2Z2Z2);
        mul_mod_device(&S2, &Q->Y, &Z1Z1Z1);
        
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
        } else {
            double_point_jac(R, P);
        }
        return;
    }
    
    
    mul_mod_device(&temp, &Z2Z2, &Q->Z);
    mul_mod_device(&S1, &P->Y, &temp);
    
    mul_mod_device(&temp, &Z1Z1, &P->Z);
    mul_mod_device(&S2, &Q->Y, &temp);
    
    
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.cc.u32 %7, %15, %23;\n\t"
        : "=r"(R_big.data[0]), "=r"(R_big.data[1]), "=r"(R_big.data[2]), "=r"(R_big.data[3]),
          "=r"(R_big.data[4]), "=r"(R_big.data[5]), "=r"(R_big.data[6]), "=r"(R_big.data[7])
        : "r"(S2.data[0]), "r"(S2.data[1]), "r"(S2.data[2]), "r"(S2.data[3]),
          "r"(S2.data[4]), "r"(S2.data[5]), "r"(S2.data[6]), "r"(S2.data[7]),
          "r"(S1.data[0]), "r"(S1.data[1]), "r"(S1.data[2]), "r"(S1.data[3]),
          "r"(S1.data[4]), "r"(S1.data[5]), "r"(S1.data[6]), "r"(S1.data[7])
    );
    
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
    if (borrow) {
        asm volatile(
            "add.cc.u32 %0, %0, %8;\n\t"
            "addc.cc.u32 %1, %1, %9;\n\t"
            "addc.cc.u32 %2, %2, %10;\n\t"
            "addc.cc.u32 %3, %3, %11;\n\t"
            "addc.cc.u32 %4, %4, %12;\n\t"
            "addc.cc.u32 %5, %5, %13;\n\t"
            "addc.cc.u32 %6, %6, %14;\n\t"
            "addc.u32 %7, %7, %15;\n\t"
            : "+r"(R_big.data[0]), "+r"(R_big.data[1]), "+r"(R_big.data[2]), "+r"(R_big.data[3]),
              "+r"(R_big.data[4]), "+r"(R_big.data[5]), "+r"(R_big.data[6]), "+r"(R_big.data[7])
            : "r"(const_p.data[0]), "r"(const_p.data[1]), "r"(const_p.data[2]), "r"(const_p.data[3]),
              "r"(const_p.data[4]), "r"(const_p.data[5]), "r"(const_p.data[6]), "r"(const_p.data[7])
        );
    }
    
    
    mul_mod_device(&HH, &H, &H);
    mul_mod_device(&HHH, &HH, &H);
    
    
    mul_mod_device(&U2, &U1, &HH);  
    
    
    mul_mod_device(&R->X, &R_big, &R_big);
    
    
    asm volatile(
        "sub.cc.u32 %0, %0, %8;\n\t"
        "subc.cc.u32 %1, %1, %9;\n\t"
        "subc.cc.u32 %2, %2, %10;\n\t"
        "subc.cc.u32 %3, %3, %11;\n\t"
        "subc.cc.u32 %4, %4, %12;\n\t"
        "subc.cc.u32 %5, %5, %13;\n\t"
        "subc.cc.u32 %6, %6, %14;\n\t"
        "subc.cc.u32 %7, %7, %15;\n\t"
        : "+r"(R->X.data[0]), "+r"(R->X.data[1]), "+r"(R->X.data[2]), "+r"(R->X.data[3]),
          "+r"(R->X.data[4]), "+r"(R->X.data[5]), "+r"(R->X.data[6]), "+r"(R->X.data[7])
        : "r"(HHH.data[0]), "r"(HHH.data[1]), "r"(HHH.data[2]), "r"(HHH.data[3]),
          "r"(HHH.data[4]), "r"(HHH.data[5]), "r"(HHH.data[6]), "r"(HHH.data[7])
    );
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
    if (borrow) {
        ptx_u256Add(&R->X, &R->X, &const_p);
    }
    
    
    asm volatile(
        "sub.cc.u32 %0, %0, %8;\n\t"
        "subc.cc.u32 %1, %1, %9;\n\t"
        "subc.cc.u32 %2, %2, %10;\n\t"
        "subc.cc.u32 %3, %3, %11;\n\t"
        "subc.cc.u32 %4, %4, %12;\n\t"
        "subc.cc.u32 %5, %5, %13;\n\t"
        "subc.cc.u32 %6, %6, %14;\n\t"
        "subc.cc.u32 %7, %7, %15;\n\t"
        : "+r"(R->X.data[0]), "+r"(R->X.data[1]), "+r"(R->X.data[2]), "+r"(R->X.data[3]),
          "+r"(R->X.data[4]), "+r"(R->X.data[5]), "+r"(R->X.data[6]), "+r"(R->X.data[7])
        : "r"(U2.data[0]), "r"(U2.data[1]), "r"(U2.data[2]), "r"(U2.data[3]),
          "r"(U2.data[4]), "r"(U2.data[5]), "r"(U2.data[6]), "r"(U2.data[7])
    );
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
    if (borrow) {
        ptx_u256Add(&R->X, &R->X, &const_p);
    }
    
    
    asm volatile(
        "sub.cc.u32 %0, %0, %8;\n\t"
        "subc.cc.u32 %1, %1, %9;\n\t"
        "subc.cc.u32 %2, %2, %10;\n\t"
        "subc.cc.u32 %3, %3, %11;\n\t"
        "subc.cc.u32 %4, %4, %12;\n\t"
        "subc.cc.u32 %5, %5, %13;\n\t"
        "subc.cc.u32 %6, %6, %14;\n\t"
        "subc.cc.u32 %7, %7, %15;\n\t"
        : "+r"(R->X.data[0]), "+r"(R->X.data[1]), "+r"(R->X.data[2]), "+r"(R->X.data[3]),
          "+r"(R->X.data[4]), "+r"(R->X.data[5]), "+r"(R->X.data[6]), "+r"(R->X.data[7])
        : "r"(U2.data[0]), "r"(U2.data[1]), "r"(U2.data[2]), "r"(U2.data[3]),
          "r"(U2.data[4]), "r"(U2.data[5]), "r"(U2.data[6]), "r"(U2.data[7])
    );
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
    if (borrow) {
        ptx_u256Add(&R->X, &R->X, &const_p);
    }
    
    
    
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.cc.u32 %7, %15, %23;\n\t"
        : "=r"(temp.data[0]), "=r"(temp.data[1]), "=r"(temp.data[2]), "=r"(temp.data[3]),
          "=r"(temp.data[4]), "=r"(temp.data[5]), "=r"(temp.data[6]), "=r"(temp.data[7])
        : "r"(U2.data[0]), "r"(U2.data[1]), "r"(U2.data[2]), "r"(U2.data[3]),
          "r"(U2.data[4]), "r"(U2.data[5]), "r"(U2.data[6]), "r"(U2.data[7]),
          "r"(R->X.data[0]), "r"(R->X.data[1]), "r"(R->X.data[2]), "r"(R->X.data[3]),
          "r"(R->X.data[4]), "r"(R->X.data[5]), "r"(R->X.data[6]), "r"(R->X.data[7])
    );
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
    if (borrow) {
        ptx_u256Add(&temp, &temp, &const_p);
    }
    
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&S1, &S1, &HHH);
    
    
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.cc.u32 %7, %15, %23;\n\t"
        : "=r"(R->Y.data[0]), "=r"(R->Y.data[1]), "=r"(R->Y.data[2]), "=r"(R->Y.data[3]),
          "=r"(R->Y.data[4]), "=r"(R->Y.data[5]), "=r"(R->Y.data[6]), "=r"(R->Y.data[7])
        : "r"(temp.data[0]), "r"(temp.data[1]), "r"(temp.data[2]), "r"(temp.data[3]),
          "r"(temp.data[4]), "r"(temp.data[5]), "r"(temp.data[6]), "r"(temp.data[7]),
          "r"(S1.data[0]), "r"(S1.data[1]), "r"(S1.data[2]), "r"(S1.data[3]),
          "r"(S1.data[4]), "r"(S1.data[5]), "r"(S1.data[6]), "r"(S1.data[7])
    );
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));
    if (borrow) {
        ptx_u256Add(&R->Y, &R->Y, &const_p);
    }
    
    mul_mod_device(&R->Z, &P->Z, &Q->Z);
    mul_mod_device(&R->Z, &R->Z, &H);
    
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
    return __funnelshift_rc(x, x, n);
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

__device__ __forceinline__ void sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    
    uint32_t h0 = 0x6a09e667ul;
    uint32_t h1 = 0xbb67ae85ul;
    uint32_t h2 = 0x3c6ef372ul;
    uint32_t h3 = 0xa54ff53aul;
    uint32_t h4 = 0x510e527ful;
    uint32_t h5 = 0x9b05688cul;
    uint32_t h6 = 0x1f83d9abul;
    uint32_t h7 = 0x5be0cd19ul;

    uint32_t a = h0;
    uint32_t b = h1;
    uint32_t c = h2;
    uint32_t d = h3;
    uint32_t e = h4;
    uint32_t f = h5;
    uint32_t g = h6;
    uint32_t h = h7;

    uint32_t w[16];

    for (int i = 0; i < 16; i++) {
        int off = i * 4;
        uint32_t val = 0;
        
        if (off < len) val |= ((uint32_t)data[off]) << 24;
        if (off + 1 < len) val |= ((uint32_t)data[off + 1]) << 16;
        if (off + 2 < len) val |= ((uint32_t)data[off + 2]) << 8;
        if (off + 3 < len) val |= ((uint32_t)data[off + 3]);
        
        if (off <= len && len < off + 4) {
            int pad_pos = len - off;
            val |= 0x80u << (24 - pad_pos * 8);
        }
        
        if (i == 14) {
            val = 0;
        }
        if (i == 15) {
            val = len * 8;
        }
        
        w[i] = val;
    }

    #define ROUND(wi, ki) { \
        uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + ki + wi; \
        uint32_t t2 = Sigma0(a) + Maj(a, b, c); \
        h = g; g = f; f = e; e = d + t1; \
        d = c; c = b; b = a; a = t1 + t2; \
    }

    ROUND(w[0], c_K[0]); ROUND(w[1], c_K[1]); ROUND(w[2], c_K[2]); ROUND(w[3], c_K[3]);
    ROUND(w[4], c_K[4]); ROUND(w[5], c_K[5]); ROUND(w[6], c_K[6]); ROUND(w[7], c_K[7]);
    ROUND(w[8], c_K[8]); ROUND(w[9], c_K[9]); ROUND(w[10], c_K[10]); ROUND(w[11], c_K[11]);
    ROUND(w[12], c_K[12]); ROUND(w[13], c_K[13]); ROUND(w[14], c_K[14]); ROUND(w[15], c_K[15]);

    #define EXTEND_ROUND(i) { \
        uint32_t s0 = sigma0(w[(i-15) & 15]); \
        uint32_t s1 = sigma1(w[(i-2) & 15]); \
        w[i & 15] = w[(i-16) & 15] + s0 + w[(i-7) & 15] + s1; \
        ROUND(w[i & 15], c_K[i]); \
    }

    EXTEND_ROUND(16); EXTEND_ROUND(17); EXTEND_ROUND(18); EXTEND_ROUND(19);
    EXTEND_ROUND(20); EXTEND_ROUND(21); EXTEND_ROUND(22); EXTEND_ROUND(23);
    EXTEND_ROUND(24); EXTEND_ROUND(25); EXTEND_ROUND(26); EXTEND_ROUND(27);
    EXTEND_ROUND(28); EXTEND_ROUND(29); EXTEND_ROUND(30); EXTEND_ROUND(31);
    EXTEND_ROUND(32); EXTEND_ROUND(33); EXTEND_ROUND(34); EXTEND_ROUND(35);
    EXTEND_ROUND(36); EXTEND_ROUND(37); EXTEND_ROUND(38); EXTEND_ROUND(39);
    EXTEND_ROUND(40); EXTEND_ROUND(41); EXTEND_ROUND(42); EXTEND_ROUND(43);
    EXTEND_ROUND(44); EXTEND_ROUND(45); EXTEND_ROUND(46); EXTEND_ROUND(47);
    EXTEND_ROUND(48); EXTEND_ROUND(49); EXTEND_ROUND(50); EXTEND_ROUND(51);
    EXTEND_ROUND(52); EXTEND_ROUND(53); EXTEND_ROUND(54); EXTEND_ROUND(55);
    EXTEND_ROUND(56); EXTEND_ROUND(57); EXTEND_ROUND(58); EXTEND_ROUND(59);
    EXTEND_ROUND(60); EXTEND_ROUND(61); EXTEND_ROUND(62); EXTEND_ROUND(63);

    h0 += a; h1 += b; h2 += c; h3 += d; h4 += e; h5 += f; h6 += g; h7 += h;

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
#define R2(aL,bL,cL,dL,eL,fL,xL,sL,kL, aR,bR,cR,dR,eR,fR,xR,sR,kR) \
	{ \
		uint32_t tL = aL + fL(bL,cL,dL) + xL + kL; \
		uint32_t tR = aR + fR(bR,cR,dR) + xR + kR; \
		uint32_t cL10 = __funnelshift_lc(cL,cL,10); \
		uint32_t cR10 = __funnelshift_lc(cR,cR,10); \
		aL = eL; aR = eR; \
		eL = dL; eR = dR; \
		dL = cL10; dR = cR10; \
		cL = bL; cR = bR; \
		bL = __funnelshift_lc(tL,tL,sL) + aL; \
		bR = __funnelshift_lc(tR,tR,sR) + aR; \
	}


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

__device__ void ripemd160(const uint8_t* __restrict__ msg, 
                          uint8_t* __restrict__ out) {
    
    const uint32_t* msg32 = reinterpret_cast<const uint32_t*>(msg);
    
    uint32_t X0 = msg32[0], X1 = msg32[1], X2 = msg32[2], X3 = msg32[3];
    uint32_t X4 = msg32[4], X5 = msg32[5], X6 = msg32[6], X7 = msg32[7];
    uint32_t X8 = 0x80, X9 = 0, X10 = 0, X11 = 0;
    uint32_t X12 = 0, X13 = 0, X14 = 256, X15 = 0;
    
    uint32_t AL = 0x67452301, BL = 0xEFCDAB89, CL = 0x98BADCFE;
    uint32_t DL = 0x10325476, EL = 0xC3D2E1F0;
    uint32_t AR = 0x67452301, BR = 0xEFCDAB89, CR = 0x98BADCFE;
    uint32_t DR = 0x10325476, ER = 0xC3D2E1F0;
    
    R2(AL,BL,CL,DL,EL,F,X0,11,0, AR,BR,CR,DR,ER,J,X5,8,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X1,14,0, AR,BR,CR,DR,ER,J,X14,9,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X2,15,0, AR,BR,CR,DR,ER,J,X7,9,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X3,12,0, AR,BR,CR,DR,ER,J,X0,11,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X4,5,0, AR,BR,CR,DR,ER,J,X9,13,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X5,8,0, AR,BR,CR,DR,ER,J,X2,15,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X6,7,0, AR,BR,CR,DR,ER,J,X11,15,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X7,9,0, AR,BR,CR,DR,ER,J,X4,5,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X8,11,0, AR,BR,CR,DR,ER,J,X13,7,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X9,13,0, AR,BR,CR,DR,ER,J,X6,7,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X10,14,0, AR,BR,CR,DR,ER,J,X15,8,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X11,15,0, AR,BR,CR,DR,ER,J,X8,11,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X12,6,0, AR,BR,CR,DR,ER,J,X1,14,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X13,7,0, AR,BR,CR,DR,ER,J,X10,14,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X14,9,0, AR,BR,CR,DR,ER,J,X3,12,0x50A28BE6);
    R2(AL,BL,CL,DL,EL,F,X15,8,0, AR,BR,CR,DR,ER,J,X12,6,0x50A28BE6);
    
    R2(AL,BL,CL,DL,EL,G,X7,7,0x5A827999, AR,BR,CR,DR,ER,I,X6,9,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X4,6,0x5A827999, AR,BR,CR,DR,ER,I,X11,13,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X13,8,0x5A827999, AR,BR,CR,DR,ER,I,X3,15,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X1,13,0x5A827999, AR,BR,CR,DR,ER,I,X7,7,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X10,11,0x5A827999, AR,BR,CR,DR,ER,I,X0,12,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X6,9,0x5A827999, AR,BR,CR,DR,ER,I,X13,8,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X15,7,0x5A827999, AR,BR,CR,DR,ER,I,X5,9,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X3,15,0x5A827999, AR,BR,CR,DR,ER,I,X10,11,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X12,7,0x5A827999, AR,BR,CR,DR,ER,I,X14,7,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X0,12,0x5A827999, AR,BR,CR,DR,ER,I,X15,7,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X9,15,0x5A827999, AR,BR,CR,DR,ER,I,X8,12,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X5,9,0x5A827999, AR,BR,CR,DR,ER,I,X12,7,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X2,11,0x5A827999, AR,BR,CR,DR,ER,I,X4,6,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X14,7,0x5A827999, AR,BR,CR,DR,ER,I,X9,15,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X11,13,0x5A827999, AR,BR,CR,DR,ER,I,X1,13,0x5C4DD124);
    R2(AL,BL,CL,DL,EL,G,X8,12,0x5A827999, AR,BR,CR,DR,ER,I,X2,11,0x5C4DD124);
    
    R2(AL,BL,CL,DL,EL,H,X3,11,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X15,9,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X10,13,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X5,7,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X14,6,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X1,15,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X4,7,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X3,11,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X9,14,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X7,8,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X15,9,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X14,6,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X8,13,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X6,6,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X1,15,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X9,14,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X2,14,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X11,12,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X7,8,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X8,13,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X0,13,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X12,5,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X6,6,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X2,14,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X13,5,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X10,13,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X11,12,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X0,13,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X5,7,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X4,7,0x6D703EF3);
    R2(AL,BL,CL,DL,EL,H,X12,5,0x6ED9EBA1, AR,BR,CR,DR,ER,H,X13,5,0x6D703EF3);
    
    R2(AL,BL,CL,DL,EL,I,X1,11,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X8,15,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X9,12,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X6,5,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X11,14,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X4,8,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X10,15,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X1,11,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X0,14,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X3,14,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X8,15,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X11,14,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X12,9,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X15,6,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X4,8,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X0,14,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X13,9,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X5,6,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X3,14,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X12,9,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X7,5,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X2,12,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X15,6,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X13,9,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X14,8,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X9,12,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X5,6,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X7,5,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X6,5,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X10,15,0x7A6D76E9);
    R2(AL,BL,CL,DL,EL,I,X2,12,0x8F1BBCDC, AR,BR,CR,DR,ER,G,X14,8,0x7A6D76E9);
    
    R2(AL,BL,CL,DL,EL,J,X4,9,0xA953FD4E, AR,BR,CR,DR,ER,F,X12,8,0);
    R2(AL,BL,CL,DL,EL,J,X0,15,0xA953FD4E, AR,BR,CR,DR,ER,F,X15,5,0);
    R2(AL,BL,CL,DL,EL,J,X5,5,0xA953FD4E, AR,BR,CR,DR,ER,F,X10,12,0);
    R2(AL,BL,CL,DL,EL,J,X9,11,0xA953FD4E, AR,BR,CR,DR,ER,F,X4,9,0);
    R2(AL,BL,CL,DL,EL,J,X7,6,0xA953FD4E, AR,BR,CR,DR,ER,F,X1,12,0);
    R2(AL,BL,CL,DL,EL,J,X12,8,0xA953FD4E, AR,BR,CR,DR,ER,F,X5,5,0);
    R2(AL,BL,CL,DL,EL,J,X2,13,0xA953FD4E, AR,BR,CR,DR,ER,F,X8,14,0);
    R2(AL,BL,CL,DL,EL,J,X10,12,0xA953FD4E, AR,BR,CR,DR,ER,F,X7,6,0);
    R2(AL,BL,CL,DL,EL,J,X14,5,0xA953FD4E, AR,BR,CR,DR,ER,F,X6,8,0);
    R2(AL,BL,CL,DL,EL,J,X1,12,0xA953FD4E, AR,BR,CR,DR,ER,F,X2,13,0);
    R2(AL,BL,CL,DL,EL,J,X3,13,0xA953FD4E, AR,BR,CR,DR,ER,F,X13,6,0);
    R2(AL,BL,CL,DL,EL,J,X8,14,0xA953FD4E, AR,BR,CR,DR,ER,F,X14,5,0);
    R2(AL,BL,CL,DL,EL,J,X11,11,0xA953FD4E, AR,BR,CR,DR,ER,F,X0,15,0);
    R2(AL,BL,CL,DL,EL,J,X6,8,0xA953FD4E, AR,BR,CR,DR,ER,F,X3,13,0);
    R2(AL,BL,CL,DL,EL,J,X15,5,0xA953FD4E, AR,BR,CR,DR,ER,F,X9,11,0);
    R2(AL,BL,CL,DL,EL,J,X13,6,0xA953FD4E, AR,BR,CR,DR,ER,F,X11,11,0);
    
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
    
    
    
    for (int i = 0; i < 20; i++) {
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
    
    
    for (int window = NUM_BASE_POINTS - 1; window >= 0; window--) {
        int bit_index = window * WINDOW_SIZE;
        uint32_t word_idx = bit_index >> 5;  
        uint32_t bit_offset = bit_index & 31;
        
        uint32_t window_val;
        
        
        if (bit_offset + WINDOW_SIZE <= 32) {
            asm("bfe.u32 %0, %1, %2, %3;" 
                : "=r"(window_val) 
                : "r"(scalar->data[word_idx]), "r"(bit_offset), "r"(WINDOW_SIZE));
        } else {
            uint32_t lo = scalar->data[word_idx];
            uint32_t hi = scalar->data[word_idx + 1];
            uint32_t combined;
            
            asm("shf.r.wrap.b32 %0, %1, %2, %3;" 
                : "=r"(combined) 
                : "r"(lo), "r"(hi), "r"(bit_offset));
            
            asm("bfe.u32 %0, %1, 0, %2;" 
                : "=r"(window_val) 
                : "r"(combined), "r"(WINDOW_SIZE));
        }
        
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
    
    
    for (int window = first_window - 1; window >= 0; window--) {
        int bit_index = window * WINDOW_SIZE;
        uint32_t word_idx = bit_index >> 5;
        uint32_t bit_offset = bit_index & 31;
        
        uint32_t window_val;
        
        if (bit_offset + WINDOW_SIZE <= 32) {
            asm("bfe.u32 %0, %1, %2, %3;" 
                : "=r"(window_val) 
                : "r"(scalar->data[word_idx]), "r"(bit_offset), "r"(WINDOW_SIZE));
        } else {
            uint32_t lo = scalar->data[word_idx];
            uint32_t hi = scalar->data[word_idx + 1];
            uint32_t combined;
            
            asm("shf.r.wrap.b32 %0, %1, %2, %3;" 
                : "=r"(combined) 
                : "r"(lo), "r"(hi), "r"(bit_offset));
            
            asm("bfe.u32 %0, %1, 0, %2;" 
                : "=r"(window_val) 
                : "r"(combined), "r"(WINDOW_SIZE));
        }
        
        ECPointJac temp = G_base_precomp[window][window_val];
        if (window_val != 0) {
            add_point_jac(result, result, &temp);
        }
    }
}
__device__ void jacobian_batch_to_hash160(const ECPointJac points[BATCH_SIZE], uint8_t hash160_out[BATCH_SIZE][20]) {
    
    bool is_valid[BATCH_SIZE];
    uint8_t valid_indices[BATCH_SIZE];
    uint8_t valid_count = 0;
    
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        
        uint32_t z_check;
        
        
        uint4 z_vec0 = *((uint4*)&points[i].Z.data[0]);
        uint4 z_vec1 = *((uint4*)&points[i].Z.data[4]);
        
        
        asm("{\n\t"
            ".reg .u32 t0, t1, t2, t3;\n\t"
            "or.b32 t0, %1, %2;\n\t"
            "or.b32 t1, %3, %4;\n\t"
            "or.b32 t2, %5, %6;\n\t"
            "or.b32 t3, %7, %8;\n\t"
            "or.b32 t0, t0, t1;\n\t"
            "or.b32 t2, t2, t3;\n\t"
            "or.b32 %0, t0, t2;\n\t"
            "}"
            : "=r"(z_check)
            : "r"(z_vec0.x), "r"(z_vec0.y), "r"(z_vec0.z), "r"(z_vec0.w),
              "r"(z_vec1.x), "r"(z_vec1.y), "r"(z_vec1.z), "r"(z_vec1.w));
        
        is_valid[i] = (!points[i].infinity) && (z_check != 0);
        
        if (!is_valid[i]) {
            
            uint4* zero_ptr = (uint4*)hash160_out[i];
            zero_ptr[0] = make_uint4(0, 0, 0, 0);
            zero_ptr[1] = make_uint4(0, 0, 0, 0);
            *((uint32_t*)(hash160_out[i] + 16)) = 0;
        } else {
            valid_indices[valid_count++] = i;
        }
    }
    
    if (valid_count == 0) return;
    
    
    BigInt products[BATCH_SIZE];
    BigInt inverses[BATCH_SIZE];
    
    copy_bigint(&products[0], &points[valid_indices[0]].Z);
    
    
    for (int i = 1; i < valid_count; i++) {
        mul_mod_device(&products[i], &products[i-1], &points[valid_indices[i]].Z);
    }
    
    
    BigInt current_inv;
    mod_inverse(&current_inv, &products[valid_count - 1]);
    
    
    for (int i = valid_count - 1; i > 0; i--) {
        mul_mod_device(&inverses[i], &current_inv, &products[i-1]);
        mul_mod_device(&current_inv, &current_inv, &points[valid_indices[i]].Z);
    }
    copy_bigint(&inverses[0], &current_inv);
    
    
    for (int v = 0; v < valid_count; v++) {
        uint8_t idx = valid_indices[v];
        
        
        BigInt Zinv2, Zinv3;
        mul_mod_device(&Zinv2, &inverses[v], &inverses[v]);
        mul_mod_device(&Zinv3, &Zinv2, &inverses[v]);
        
        
        BigInt x_affine, y_affine;
        mul_mod_device(&x_affine, &points[idx].X, &Zinv2);
        mul_mod_device(&y_affine, &points[idx].Y, &Zinv3);
        
        
        uint8_t pubkey[33];
        
        
        uint32_t parity;
        asm("bfe.u32 %0, %1, 0, 1;" : "=r"(parity) : "r"(y_affine.data[0]));
        pubkey[0] = 0x02 | parity;
        
        
        uint32_t* x_data = x_affine.data;
        
        for (int j = 0; j < 8; j++) {
            uint32_t word = x_data[7 - j];
            uint32_t swapped;
            
            
            asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(swapped) : "r"(word));
            
            
            *((uint32_t*)&pubkey[1 + j * 4]) = swapped;
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