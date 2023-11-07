#include <iostream>

#define checkCudaErrors(err)                                                                                          \
  {                                                                                                                   \
    if (err != cudaSuccess) {                                                                                         \
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                    \
              cudaGetErrorString(err), __FILE__, __LINE__);                                                           \
      exit(-1);                                                                                                       \
    }                                                                                                                 \
  }



void* d_buffer;
int* h_buffer;

__attribute__((constructor)) void init() {
  d_buffer = h_buffer = nullptr;
}
__attribute__((destructor)) void destroy () {
  if (h_buffer != nullptr) {
    delete[] h_buffer;
    h_buffer = nullptr;
    std::cout << "host buffer freed\n";
  }
  if (d_buffer != nullptr) {
    checkCudaErrors(cudaFree(d_buffer));
    d_buffer = nullptr;
    std::cout << "device buffer freed\n";
  }
}

void allocateAndCopy (int *src, int vol) {
  if (h_buffer == nullptr) {
    h_buffer = new int[vol];
  }
  if (d_buffer == nullptr) {
    checkCudaErrors(cudaMalloc(&d_buffer, sizeof(int) * vol));
  }

  memcpy (h_buffer, src, sizeof(int) * vol);
  checkCudaErrors(cudaMemcpy(d_buffer, src, sizeof(int) * vol, cudaMemcpyHostToDevice));
}



int main () {

  constexpr int vol = 1024;
  int arr[vol];

  for (int i = 0; i < 5; i++) {
    allocateAndCopy (arr, vol);
  }
  // if you don't use next line, just let destructor to do cudaFree, then an error will occur: checkCudaErrors() API error = 0004 "driver shutting down" from file <test_cudaFree.cu>, line 27.
  destroy();
};
