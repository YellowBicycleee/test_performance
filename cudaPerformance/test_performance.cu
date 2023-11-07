#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
// #include <trove/ptr.h>
// #define DEBUG

template<typename T>
struct Complex {
// private:
  T real_;
  T imag_;

// public:
  Complex() = default;
  __device__ __host__ Complex(T real, T imag) : real_(real), imag_(imag) {}
  __device__ __host__ Complex(const Complex& rhs) : real_(rhs.real_), imag_(rhs.imag_) {}
  // __device__ __host__ Complex& operator=(const Complex& rhs) {
  //   memcpy(this, &rhs, sizeof(Complex));
  //   return *this;
  // }
  __device__ __host__ T real() const { return real_;}
  __device__ __host__ T imag() const { return imag_;}
};


#define RED "\033[31m"
#define BLUE "\e[0;34m" 
#define CLR "\033[0m"
#define L_RED                 "\e[1;31m"  


#define checkCudaErrors(err)                                                                                           \
  {                                                                                                                    \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                                                            \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

#define BLOCK_SIZE 256
#define WARP_SIZE 32





// #define DEBUG

const int Lx = 32;
const int Ly = 32;
const int Lz = 32;
const int Lt = 64;
const int Vol = Lx * Ly * Lz * Lt;

__device__ void print_vector(double *data)
{
  // 9 * 2 ----> 6 * 3 print
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%.2lf\t", data[i * 3 + j]);
    }
    printf("\n");
  }
}

__device__ void loadVectorBySharedMemory(double *shared_buffer, const double *origin, double *src_local)
{
  // src_local is register variable
  int warp_index = threadIdx.x / WARP_SIZE; // in block

  const double *block_src = origin + blockDim.x * blockIdx.x * 9 * 2; // the start addr of block on global memory
  double *shared_src = shared_buffer + threadIdx.x * 9 * 2;           // the thread vector addr on shared memory

  // load from global memory, every WARP
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * 9 * 2; i += WARP_SIZE) {
    shared_buffer[warp_index * WARP_SIZE * 9 * 2 + i] = block_src[warp_index * WARP_SIZE * 9 * 2 + i];
  }

  // load src from shared memory
  for (int i = 0; i < 9 * 2; i++) {
    src_local[i] = shared_src[i];
  }
}

__device__ void storeVectorBySharedMemory(double *shared_buffer, double *origin, double *src_local)
{
  // src_local is register variable
  int warp_index = threadIdx.x / WARP_SIZE; // in block

  double *block_src = origin + blockDim.x * blockIdx.x * 9 * 2; // the start addr of block on global memory
  double *shared_src = shared_buffer + threadIdx.x * 9 * 2;     // the thread vector addr on shared memory

  for (int i = 0; i < 9 * 2; i++) {
    shared_src[i] = src_local[i];
  }
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * 9 * 2; i += WARP_SIZE) {
    block_src[warp_index * WARP_SIZE * 9 * 2 + i] = shared_buffer[warp_index * WARP_SIZE * 9 * 2 + i];
  }
}

__global__ void initialize(void *data)
{
  int thread = blockIdx.x * blockDim.x + threadIdx.x;

  double *ptr = static_cast<double *>(data) + thread * 9 * 2;

  for (int i = 0; i < 9; i++) {
    ptr[2 * i + 0] = thread + i;
    ptr[2 * i + 1] = thread + i;
  }
}

// 3*3*2 tanspose---complex transpose
__device__ void transpose(double *u)
{
  double temp;
  temp = u[1 * 2 + 0];
  u[1 * 2 + 0] = u[3 * 2 + 0];
  u[3 * 2 + 0] = temp;
  temp = u[1 * 2 + 1];
  u[1 * 2 + 1] = u[3 * 2 + 1];
  u[3 * 2 + 1] = temp;

  temp = u[2 * 2 + 0];
  u[2 * 2 + 0] = u[6 * 2 + 0];
  u[6 * 2 + 0] = temp;
  temp = u[2 * 2 + 1];
  u[2 * 2 + 1] = u[6 * 2 + 1];
  u[6 * 2 + 1] = temp;

  temp = u[5 * 2 + 0];
  u[5 * 2 + 0] = u[7 * 2 + 0];
  u[7 * 2 + 0] = temp;
  temp = u[5 * 2 + 1];
  u[5 * 2 + 1] = u[7 * 2 + 1];
  u[7 * 2 + 1] = temp;
}


// double2 tanspose---complex transpose
__device__ void transpose_double2(double2 *u)
{
  double2 temp;
  temp = u[1];
  u[1] = u[3];
  u[3] = temp;

  temp = u[2];
  u[2] = u[6];
  u[6] = temp;

  temp = u[5];
  u[5] = u[7];
  u[7] = temp;
}


// double2 tanspose---complex transpose
__device__ void transpose_complex(Complex<double> *u)
{
  Complex<double> temp;
  temp = u[1];
  u[1] = u[3];
  u[3] = temp;

  temp = u[2];
  u[2] = u[6];
  u[6] = temp;

  temp = u[5];
  u[5] = u[7];
  u[7] = temp;
}


__global__ void naive_transpose(void *dst, const void *src)
{
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const double *src_ptr = static_cast<const double *>(src) + thread * 9 * 2;
  double *dst_ptr = static_cast<double *>(dst) + thread * 9 * 2;

  double data_local[9 * 2];
  for (int i = 0; i < 9 * 2; i++) {
    data_local[i] = src_ptr[i];
  }

  transpose(data_local);

  for (int i = 0; i < 9 * 2; i++) {
    dst_ptr[i] = data_local[i];
  }

#ifdef DEBUG
  if (thread == 10) {
    print_vector(dst_ptr);
  }
#endif

}


__global__ void naive_transpose_nonfunction(void *dst, const void *src)
{
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const double *src_ptr = static_cast<const double *>(src) + thread * 9 * 2;
  double *dst_ptr = static_cast<double *>(dst) + thread * 9 * 2;

  double data_local[9 * 2];
  for (int i = 0; i < 9 * 2; i++) {
    data_local[i] = src_ptr[i];
  }

  // transpose(data_local);
  double temp;
  temp = data_local[1 * 2 + 0];
  data_local[1 * 2 + 0] = data_local[3 * 2 + 0];
  data_local[3 * 2 + 0] = temp;
  temp = data_local[1 * 2 + 1];
  data_local[1 * 2 + 1] = data_local[3 * 2 + 1];
  data_local[3 * 2 + 1] = temp;

  temp = data_local[2 * 2 + 0];
  data_local[2 * 2 + 0] = data_local[6 * 2 + 0];
  data_local[6 * 2 + 0] = temp;
  temp = data_local[2 * 2 + 1];
  data_local[2 * 2 + 1] = data_local[6 * 2 + 1];
  data_local[6 * 2 + 1] = temp;

  temp = data_local[5 * 2 + 0];
  data_local[5 * 2 + 0] = data_local[7 * 2 + 0];
  data_local[7 * 2 + 0] = temp;
  temp = data_local[5 * 2 + 1];
  data_local[5 * 2 + 1] = data_local[7 * 2 + 1];
  data_local[7 * 2 + 1] = temp;


  for (int i = 0; i < 9 * 2; i++) {
    dst_ptr[i] = data_local[i];
  }

#ifdef DEBUG
  if (thread == 10) {
    print_vector(dst_ptr);
  }
#endif

}



__global__ void shared_transpose(void *dst, const void *src)
{
  __shared__ double shared_buffer[BLOCK_SIZE * 9 * 2];
  // int thread = blockIdx.x * blockDim.x + threadIdx.x;
  //  load data from global memory to register
  // double* src = static_cast<double*>(data) + thread * 9 * 2;
  double data_local[9 * 2];
  loadVectorBySharedMemory(shared_buffer, static_cast<const double *>(src), data_local);
  // transpose(data_local);
  storeVectorBySharedMemory(shared_buffer, static_cast<double *>(dst), data_local);
}

__global__ void naive_copy(void *dst, const void *src)
{
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const double *src_ptr = static_cast<const double *>(src) + thread * 9 * 2;
  double *dst_ptr = static_cast<double *>(dst) + thread * 9 * 2;

  for (int i = 0; i < 9 * 2; ++i) {
    dst_ptr[i] = src_ptr[i];
  }
}

template <typename T, int n> struct S {
  T v[n];
  __host__ __device__ inline const T &operator[](int i) const { return v[i]; }
  __host__ __device__ inline T &operator[](int i) { return v[i]; }
};

// __global__ void trove_transpose(void *dst, const void *src)
// {
//   using array_t = S<double, 9 * 2>;

//   int thread = blockIdx.x * blockDim.x + threadIdx.x;
//   double data_local[9 * 2];

//   *(reinterpret_cast<array_t *>(data_local)) =
//       *(trove::coalesced_ptr<const array_t>(reinterpret_cast<const array_t *>(src) + thread));

//   // load data from global memory to register
//   // transpose
//   // transpose(data_local);
//   *(trove::coalesced_ptr<array_t>(reinterpret_cast<array_t *>(dst) + thread)) =
//       *(reinterpret_cast<array_t *>(data_local));
// }


// use these two functions to ensure the result
__device__ void print_dst_double1(void *dst) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  double *dst_ptr = static_cast<double *>(dst) + thread;

  printf(RED"");
  for (int i = 0; i < 9; i++) {
    if (i != 0 && i % 3 == 0) printf("\n");
    printf("(%lf  %lf)\t", dst_ptr[0], dst_ptr[Vol]);
    dst_ptr += 2 * Vol;
  }
  printf(CLR"\n");
}
__device__ void print_dst_double2(void *dst) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  double *dst_ptr = static_cast<double *>(dst) + thread * 2;

  printf(BLUE"");
  for (int i = 0; i < 9; i++) {
    if (i != 0 && i % 3 == 0) printf("\n");
    printf("(%lf  %lf)\t", dst_ptr[0], dst_ptr[1]);
    dst_ptr += 2 * Vol;
  }
  printf(CLR"\n");
}

__global__ void shift_to_coalesce_double1(void* dst, void* src) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const double *src_ptr = static_cast<const double *>(src) + thread * 9 * 2;
  double *dst_ptr = static_cast<double *>(dst) + thread;
  for (int i = 0; i < 9 * 2; i++) {
    *dst_ptr = src_ptr[i];
    dst_ptr += Vol;
  }
}
__global__ void shift_to_coalesce_double2(void* dst, void* src) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const double *src_ptr = static_cast<const double *>(src) + thread * 9 * 2;
  double *dst_ptr = static_cast<double *>(dst) + thread * 2;
  for (int i = 0; i < 9; i++) {
    dst_ptr[0] = src_ptr[2*i];
    dst_ptr[1] = src_ptr[2*i+1];
    dst_ptr += 2 * Vol;
  }
}


// assumption: store real and imag separately
//            every time offset add vol
__global__ void test_coalesce_double1 (void *dst, const void *src) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const double *src_ptr = static_cast<const double *>(src) + thread;
  double *dst_ptr = static_cast<double *>(dst) + thread;

  double data_local[9 * 2];
  for (int i = 0; i < 9 * 2; i++) {
    data_local[i] = *src_ptr;
    src_ptr += Vol;
  }

  transpose(data_local);

  for (int i = 0; i < 9 * 2; i++) {
    *dst_ptr = data_local[i];
    dst_ptr += Vol;
  }
#ifdef DEBUG
  if (thread == 10) {
    print_dst_double1(dst);
  }
#endif
}

// assumption: store real and imag separately
__global__ void test_coalesce_2double (void *dst, const void *src) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const double *src_ptr = static_cast<const double *>(src) + thread * 2;
  double *dst_ptr = static_cast<double *>(dst) + thread * 2;

  double data_local[9 * 2];
  for (int i = 0; i < 9; i++) {
    data_local[2*i] = src_ptr[0];
    data_local[2*i+1] = src_ptr[1];
    src_ptr += 2 * Vol;
  }

  transpose(data_local);

  for (int i = 0; i < 9; i++) {
    dst_ptr[0] = data_local[2*i];
    dst_ptr[1] = data_local[2*i+1];
    dst_ptr += 2 * Vol;
  }

#ifdef DEBUG
  if (thread == 10) {
    print_dst_double2(dst);
  }
#endif
}



// assumption: store real and imag separately
__global__ void test_coalesce_double2 (void *dst, const void *src) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const double2 *src_ptr = static_cast<const double2 *>(src) + thread;
  double2 *dst_ptr = static_cast<double2 *>(dst) + thread;

  double2 data_local[9];
  for (int i = 0; i < 9; i++) {
    data_local[i] = *src_ptr;
    src_ptr += Vol;
  }

  // transpose(reinterpret_cast<double*>(data_local));

  transpose_double2(data_local);
  for (int i = 0; i < 9; i++) {
    *dst_ptr = data_local[i];
    dst_ptr += Vol;
  }

#ifdef DEBUG
  if (thread == 10) {
    print_dst_double2(dst);
  }
#endif
}



// assumption: store real and imag separately
__global__ void test_coalesce_complex (void *dst, const void *src) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const Complex<double> *src_ptr = static_cast<const Complex<double> *>(src) + thread;
  Complex<double> *dst_ptr = static_cast<Complex<double> *>(dst) + thread;

  Complex<double> data_local[9];
  for (int i = 0; i < 9; i++) {
    data_local[i] = *src_ptr;
    src_ptr += Vol;
  }

  // transpose(reinterpret_cast<double*>(data_local));

  // transpose_double2(data_local);
  // transpose_complex(Complex<double> *u)
  transpose_complex(data_local);
  for (int i = 0; i < 9; i++) {
    *dst_ptr = data_local[i];
    dst_ptr += Vol;
  }

#ifdef DEBUG
  if (thread == 10) {
    print_dst_double2(dst);
  }
#endif
}



void compare()
{
  // void* h_data = (void*)malloc(Vol * sizeof(double) * 9 * 2);
  void *d_src;
  void *d_dst;

  checkCudaErrors(cudaMalloc(&d_src, Vol * sizeof(double) * 9 * 2));
  checkCudaErrors(cudaMalloc(&d_dst, Vol * sizeof(double) * 9 * 2));

  int grid_size = Vol / BLOCK_SIZE;
  // initialize
  initialize<<<grid_size, BLOCK_SIZE>>>(d_src);
  checkCudaErrors(cudaDeviceSynchronize());

  // end

  // naive
  auto start = std::chrono::high_resolution_clock::now();
  naive_transpose<<<grid_size, BLOCK_SIZE>>>(d_dst, d_src);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("naive loading takes %.9lf sec\n", double(duration) / 1e9);
  // naive end

  // // trove
  // start = std::chrono::high_resolution_clock::now();
  // trove_transpose<<<grid_size, BLOCK_SIZE>>>(d_dst, d_src);
  // checkCudaErrors(cudaDeviceSynchronize());
  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  // printf("trove loading takes %.9lf sec\n", double(duration) / 1e9);

  // shared
  start = std::chrono::high_resolution_clock::now();
  shared_transpose<<<grid_size, BLOCK_SIZE>>>(d_dst, d_src);
  checkCudaErrors(cudaDeviceSynchronize());
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("shared loading takes %.9lf sec\n", double(duration) / 1e9);
  // shared end

  // shared
  start = std::chrono::high_resolution_clock::now();
  naive_copy<<<grid_size, BLOCK_SIZE>>>(d_dst, d_src);
  checkCudaErrors(cudaDeviceSynchronize());
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("naive copy takes %.9lf sec\n", double(duration) / 1e9);
  // shared end

  // free(h_data);
  checkCudaErrors(cudaFree(d_dst));
  checkCudaErrors(cudaFree(d_src));
}


void execute_double1_kernel(void* dst, void* src, int block_size, int grid_size) {

  auto start = std::chrono::high_resolution_clock::now();
  test_coalesce_double1<<<grid_size, block_size>>> (dst, src);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("double1 coalescing takes %.9lf sec\n", double(duration) / 1e9);
}
void execute_2double_kernel(void* dst, void* src, int block_size, int grid_size) {
  auto start = std::chrono::high_resolution_clock::now();
  test_coalesce_2double<<<grid_size, block_size>>> (dst, src);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("2 double coalescing takes %.9lf sec\n", double(duration) / 1e9);
}


void execute_double2_kernel(void* dst, void* src, int block_size, int grid_size) {
  auto start = std::chrono::high_resolution_clock::now();
  test_coalesce_double2<<<grid_size, block_size>>> (dst, src);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("double2 coalescing takes %.9lf sec\n", double(duration) / 1e9);
}


void execute_complex_kernel(void* dst, void* src, int block_size, int grid_size) {
  auto start = std::chrono::high_resolution_clock::now();
  test_coalesce_complex<<<grid_size, block_size>>> (dst, src);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("complex coalescing takes %.9lf sec\n", double(duration) / 1e9);
}

void execute_naive_kernel(void* dst, void* src, int block_size, int grid_size) {
  auto start = std::chrono::high_resolution_clock::now();
  naive_transpose<<<grid_size, block_size>>> (dst, src);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("naive takes %.9lf sec\n", double(duration) / 1e9);
}

void execute_naive_nonfunction_kernel(void* dst, void* src, int block_size, int grid_size) {
  auto start = std::chrono::high_resolution_clock::now();
  naive_transpose_nonfunction<<<grid_size, block_size>>> (dst, src);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("naive without function takes %.9lf sec\n", double(duration) / 1e9);
}

void compare_coalesce_double1_double2() {
  void* origin_src;
  void* naive_dst;
  void* double1_src;
  void* double2_src;
  void* two_double_src;
  void* complex_src;

  void* double1_dst;
  void* double2_dst;
  void* two_double_dst;
  void* complex_dst;

  checkCudaErrors(cudaMalloc(&origin_src, sizeof(double) * Vol * 9 * 2));
  checkCudaErrors(cudaMalloc(&naive_dst, sizeof(double) * Vol * 9 * 2));
  checkCudaErrors(cudaMalloc(&double1_src, sizeof(double) * Vol * 9 * 2));
  checkCudaErrors(cudaMalloc(&double2_src, sizeof(double) * Vol * 9 * 2));
  checkCudaErrors(cudaMalloc(&two_double_src, sizeof(double) * Vol * 9 * 2));
  checkCudaErrors(cudaMalloc(&complex_src, sizeof(double) * Vol * 9 * 2));

  checkCudaErrors(cudaMalloc(&double1_dst, sizeof(double) * Vol * 9 * 2));
  checkCudaErrors(cudaMalloc(&double2_dst, sizeof(double) * Vol * 9 * 2));
  checkCudaErrors(cudaMalloc(&two_double_dst, sizeof(double) * Vol * 9 * 2));
  checkCudaErrors(cudaMalloc(&complex_dst, sizeof(double) * Vol * 9 * 2));

  int block_size = BLOCK_SIZE;
  int grid_size = (Vol + block_size - 1) / block_size;
  // prepare data
  initialize<<<grid_size, block_size>>>(origin_src);
  checkCudaErrors(cudaDeviceSynchronize());

  shift_to_coalesce_double1<<<grid_size, block_size>>>(double1_src, origin_src);
  checkCudaErrors(cudaDeviceSynchronize());

  shift_to_coalesce_double2<<<grid_size, block_size>>>(double2_src, origin_src);
  checkCudaErrors(cudaDeviceSynchronize());

  shift_to_coalesce_double2<<<grid_size, block_size>>>(two_double_src, origin_src);
  checkCudaErrors(cudaDeviceSynchronize());

  shift_to_coalesce_double2<<<grid_size, block_size>>>(complex_src, origin_src);
  checkCudaErrors(cudaDeviceSynchronize());
  // end

  // compare
  execute_double1_kernel(double1_dst, double1_src, block_size, grid_size);
  execute_2double_kernel(two_double_dst, two_double_src, block_size, grid_size);
  execute_double2_kernel(double2_dst, double2_src, block_size, grid_size);
  execute_naive_kernel(naive_dst, origin_src, block_size, grid_size);
  execute_naive_nonfunction_kernel(naive_dst, origin_src, block_size, grid_size);
  execute_complex_kernel(complex_dst, complex_src, block_size, grid_size);


  checkCudaErrors(cudaFree(origin_src));
  checkCudaErrors(cudaFree(naive_dst));
  checkCudaErrors(cudaFree(double1_src));
  checkCudaErrors(cudaFree(double2_src));
  checkCudaErrors(cudaFree(two_double_src));
  checkCudaErrors(cudaFree(complex_src));

  checkCudaErrors(cudaFree(double1_dst));
  checkCudaErrors(cudaFree(double2_dst));
  checkCudaErrors(cudaFree(two_double_dst));
  checkCudaErrors(cudaFree(complex_dst));
}

int main()
{
  for (int i = 0; i < 3; ++i) {
    // compare();
    compare_coalesce_double1_double2();
    printf("=======================\n");
  }
  return 0;
}