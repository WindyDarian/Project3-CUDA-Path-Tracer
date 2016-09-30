#include "efficient.h"

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace StreamCompaction {
namespace Efficient {

// using StreamCompaction::Common::PerformanceTimer;
// PerformanceTimer& timer()
// {
//     // not thread-safe
//     static PerformanceTimer timer;
//     return timer;
// }

// DONE: __global__

// optimized: only launch necessary amount of threads in host code
__global__ void kernScanUpSweepPass(int max_thread_index, int add_distance, int* buffer)
{
    auto tindex = threadIdx.x + blockIdx.x * blockDim.x;

    if (tindex >= max_thread_index) { return; }

    // I encountered overflow problem with index < N here so I changed to tindex < max_thread_index
    size_t index = (add_distance * 2) * (1 + tindex) - 1;

    buffer[index] = buffer[index] + buffer[index - add_distance];
}

// optimized: only launch necessary amount of threads in host code
__global__ void kernScanDownSweepPass(int max_thread_index, int distance, int* buffer)
{
    auto tindex = threadIdx.x + blockIdx.x * blockDim.x;

    if (tindex >= max_thread_index) { return; }

    size_t index = (distance * 2) * (1 + tindex) - 1;

    auto temp = buffer[index - distance];
    buffer[index - distance] = buffer[index];
    buffer[index] = temp + buffer[index];
}

/**
* Swap value of two array members
* Used for inclusive scan as I misunderstood the requirement
*/
__global__ void kernSwap(int index1, int index2, int* buffer)
{
    auto thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index == 0)
    {
        buffer[index1] ^= buffer[index2];
        buffer[index2] ^= buffer[index1];
        buffer[index1] ^= buffer[index2];
    }
}

/**
* Set value of an array member to zero
*/
__global__ void kernSetZero(int index, int* buffer)
{
    auto thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index == 0)
    {
        buffer[index] = 0;
    }
}


int getUpSweepBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernScanUpSweepPass);
    }
    return block_size;
}

int getDownSweepBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernScanDownSweepPass);
    }
    return block_size;
}

enum class ScanType{inclusive, exclusive};

void scanInPlaceDevice(int extended_n, int* dev_buffer)
{
    auto block_size_up = getUpSweepBlockSize();
    auto full_blocks_per_grid_up = fullBlocksPerGrid(extended_n, block_size_up);
    auto block_size_down = getDownSweepBlockSize();
    auto full_blocks_per_grid_down = fullBlocksPerGrid(extended_n, block_size_down);

    // up sweep
    auto pass_count = ilog2ceil(extended_n) - 1;
    auto max_thread_index = extended_n;
    for (int d = 0; d <= pass_count; d++)
    {
        max_thread_index = extended_n >> (d + 1);
        full_blocks_per_grid_up = fullBlocksPerGrid(max_thread_index, block_size_up); // only launch needed threads
        kernScanUpSweepPass << <full_blocks_per_grid_up, block_size_up >> >(max_thread_index, 1 << d, dev_buffer);
    }

    // set the last element to zero
    kernSetZero <<<1, 1 >>>(extended_n - 1, dev_buffer);


    // down sweep
    for (int d = pass_count; d >= 0; d--)
    {
        max_thread_index = extended_n >> (d + 1);
        full_blocks_per_grid_down = fullBlocksPerGrid(max_thread_index, block_size_down); // only launch needed threads
        kernScanDownSweepPass <<<full_blocks_per_grid_down, block_size_down >>>(max_thread_index, 1 << d, dev_buffer);
    }
}

void scan_implemention(int n, int *odata, const int *idata, ScanType scan_type)
{
    if (n <= 0) { return; }

    // DONE
    // round n up to power of two
    auto extended_n = std::size_t(1) << ilog2ceil(n);
    // plus one for
    auto final_buffer_length =
        (scan_type == ScanType::inclusive) ? extended_n + 1: extended_n;

    int* dev_buffer;
    cudaMalloc((void**)&dev_buffer, final_buffer_length * sizeof(*dev_buffer));
    checkCUDAError("cudaMalloc dev_buffer failed!");

    // fill zero and copy to device buffer
    cudaMemset(dev_buffer, 0, final_buffer_length * sizeof(*idata));
    cudaMemcpy(dev_buffer, idata, n * sizeof(*idata), cudaMemcpyHostToDevice);

    auto block_size_up = getUpSweepBlockSize();
    auto full_blocks_per_grid_up = fullBlocksPerGrid(extended_n, block_size_up);
    auto block_size_down = getDownSweepBlockSize();
    auto full_blocks_per_grid_down = fullBlocksPerGrid(extended_n, block_size_down);

    // timer().startGpuTimer();

    // up sweep
    auto pass_count = ilog2ceil(extended_n) - 1;
    auto max_thread_index = extended_n;
    for (int d = 0; d <= pass_count; d++)
    {
        max_thread_index = extended_n >> (d + 1);
        full_blocks_per_grid_up = fullBlocksPerGrid(max_thread_index, block_size_up); // only launch needed threads
        kernScanUpSweepPass << <full_blocks_per_grid_up, block_size_up >> >(max_thread_index, 1 << d, dev_buffer);
    }

    if (scan_type == ScanType::inclusive)
    {
        // swap the last element of up sweep result and the real last element (0)
        kernSwap <<<1, 1>>>(extended_n - 1, final_buffer_length - 1, dev_buffer);
    }
    else
    {
        // set the last element to zero
        kernSetZero <<<1, 1>>>(extended_n - 1, dev_buffer);
    }

    // down sweep
    for (int d = pass_count; d >= 0; d--)
    {
        max_thread_index = extended_n >> (d + 1);
        full_blocks_per_grid_down = fullBlocksPerGrid(max_thread_index, block_size_down); // only launch needed threads
        kernScanDownSweepPass <<<full_blocks_per_grid_down, block_size_down >>>(max_thread_index, 1 << d, dev_buffer);
    }

    // timer().endGpuTimer();

    if (scan_type == ScanType::inclusive)
    {
        // copy with offset to make it an inclusive scan
        cudaMemcpy(odata, dev_buffer + 1, n * sizeof(*odata), cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(odata, dev_buffer, n * sizeof(*odata), cudaMemcpyDeviceToHost);
    }

    cudaFree(dev_buffer);
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 * This just call scan_implementaion
 */
void scan(int n, int *odata, const int *idata)
{
    scan_implemention(n, odata, idata, ScanType::exclusive);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    if (n <= 0) { return 0; }

    // DONE
    int* dev_idata;
    cudaMalloc((void**)&dev_idata, n * sizeof(*dev_idata));
    checkCUDAError("cudaMalloc dev_idata failed!");
    cudaMemcpy(dev_idata, idata, n * sizeof(*idata), cudaMemcpyHostToDevice);

    auto extended_n = std::size_t(1) << ilog2ceil(n); // round up to power of two for scanning

    int* dev_bools; // TODO: I could use size_t* (but which will result in a lot of rewrite for the kernel functions)
    cudaMalloc((void**)&dev_bools, extended_n * sizeof(*dev_bools));
    checkCUDAError("cudaMalloc dev_bools failed!");
    // fill zero and copy to boolean buffer
    cudaMemset(dev_bools, 0, extended_n * sizeof(*dev_bools));

    // reuse bool buffer as indices buffer
    auto dev_indices = dev_bools;

    int* dev_odata;
    cudaMalloc((void**)&dev_odata, n * sizeof(*dev_odata));
    checkCUDAError("cudaMalloc dev_odata failed!");

    auto block_size_booleanize = Common::getMapToBooleanBlockSize();
    auto full_blocks_per_grid_booleanize = fullBlocksPerGrid(n, block_size_booleanize);
    auto block_size_scatter = Common::getScatterBlocksize();
    auto full_blocks_per_grid_scatter = fullBlocksPerGrid(n, block_size_scatter);

    // timer().startGpuTimer();

    // map to boolean
    Common::kernMapToBoolean <<<full_blocks_per_grid_booleanize, block_size_booleanize >>>(n, dev_bools, dev_idata);

    // exclusively scan the dev_bools buffer
    scanInPlaceDevice(extended_n, dev_bools);

    // scatter
    Common::kernScatter <<<full_blocks_per_grid_scatter, block_size_scatter >>>(n, dev_odata, dev_idata, dev_indices);
    // timer().endGpuTimer();

    // calculate compacted length
    using dev_indices_t = std::remove_reference<decltype(*dev_indices)>::type;
    dev_indices_t result_length;
    cudaMemcpy(&result_length, dev_indices + n - 1, sizeof(result_length), cudaMemcpyDeviceToHost);
    if (idata[n - 1])
    {
        result_length += 1;
    }

    // get compacted result
    cudaMemcpy(odata, dev_odata, result_length * sizeof(*odata), cudaMemcpyDeviceToHost);

    cudaFree(dev_idata);
    cudaFree(dev_bools); // the same buffer as dev_indices
    cudaFree(dev_odata);

    return result_length;
}

}
}
