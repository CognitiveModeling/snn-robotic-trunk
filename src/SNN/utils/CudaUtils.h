#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#ifndef NO_CUDA_ASSERT
#define cudaAssert(condition) \
    if (!(condition)) { printf("%s:%d => Assertion %s failed!\n", __FILE__, __LINE__, #condition); return; }
#else
#define cudaAssert(condition) 
#endif

#define wrapThreads(value) (int((int(value) + 31)/32) * 32)

#define sgn(x) (((x) < 0) ? -1.0 : 1.0)

/* returns a random number from the given generatopr */
static __device__ unsigned rand(unsigned *rnd) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    rnd[id] ^= rnd[id] << 13;
    rnd[id] ^= rnd[id] >> 17;
    rnd[id] ^= rnd[id] << 5;
    return rnd[id];
}

#define rand_range_01(rnd) (FloatType(rand(rnd)) / 4294967295.0)

#define rand_range_d(rnd, start, end) \
    ((FloatType(rand(rnd)) / 4294967295.0) * (FloatType(end) - FloatType(start)) + FloatType(start))

#define randexp(rnd, rate) (-1.0 * log(rand_range_d32(rnd, 0.0, 1.0)) / double(rate))
#define randexp_u(rnd, rate) unsigned(round(randexp(rnd, rate)))

/* returns normal distributed random value */
static __device__ FloatType rand128n(unsigned *rnd) {
    FloatType u = 0.0, v = 0.0, s = 0.0;

    do {
        u = rand_range_d(rnd, -1.0, 1.0);
        v = rand_range_d(rnd, -1.0, 1.0);
        s = pow(u, 2) + pow(v, 2);
    } while (s >= 1.0 || s == 0.0);

    return u * sqrt((-2 * log(s)) / s);
}

#endif /* __CUDA_UTILS_H__ */
