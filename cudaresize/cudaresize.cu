#include <stdlib.h>
#include <stdio.h>
#include <cutil.h>
#include <memcpy.cu>
#include <cubicPrefilter3D.cu>
#include <cubicTex3D.cu>

#include "cudaresize.h"

texture<float, cudaTextureType3D, cudaReadModeElementType> texRef;

__global__ 
void transformKernel(float *output, float wr, float hr, float dr, 
                     unsigned int width, unsigned int height, unsigned int depth, 
                     unsigned int zoffset) {
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    float u = ((float)x)*wr + 0.5f;
    float v = ((float)y)*hr + 0.5f;
    float w = ((float)(z + zoffset))*dr + 0.5f;

    // Read from texture and write to global memory
    if (x < width && y < height && z < depth)
        output[z*width*height + y*width + x] = tex3D(texRef, u, v, w);
}

template<class T> 
void interpolate_template(
        const T *input_array, T *output_array, unsigned int width, unsigned int height, unsigned int depth,
        unsigned int new_width, unsigned int new_height, unsigned int new_depth
    ) {
    // Load the input volume into the device memory; this also converts it to a float
    cudaPitchedPtr bsplineCoeffs = CastVolumeHostToDevice(input_array, width, height, depth);

    // Run the pre-filter
    CubicBSplinePrefilter3D((float*)bsplineCoeffs.ptr, bsplineCoeffs.pitch, width, height, depth);

    // Now, we need to copy everything into a CUDA array. Ideally, this should be done entirely on 
    // the device, but for large input volumes we may run out of GPU memory so we'll have to copy 
    // it back to host RAM and then to the GPU again

    // bsplineCoeffs.pitch is the actual size of each row, including GPU-specific padding bytes 
    // (for performance reasons). This is the total size of the memory allocated for the 
    // bsplineCoeffs array, and should be the same as required for the array
    size_t needed = bsplineCoeffs.pitch*sizeof(float)*height*depth;

    // Read the total available GPU memory
    size_t free_mem, total;
    cudaMemGetInfo(&free_mem, &total);
    printf("free_mem: %zu, total: %zu, needed: %zu\n", free_mem, total, needed);

    bool enough_gpu = free_mem < (needed + (needed >> 3));

    // Add ~12% to be safe
    if (true || !enough_gpu) {
        // Allocate a temp array on the host
        float *temp = (float *)malloc(sizeof(float)*width*height*depth);
        // Create a pitched pointer for CUDA (no padding bytes)
        cudaPitchedPtr temp_ptr = make_cudaPitchedPtr((void*)temp, width*sizeof(float), width, height);

        // Create the memcpy parameters
        cudaMemcpy3DParms p = {0};
        // Extent width is in bytes for linear memory
        p.extent = make_cudaExtent(width*sizeof(float), height, depth);
        p.srcPtr = bsplineCoeffs;
        p.dstPtr = temp_ptr;
        p.kind = cudaMemcpyDeviceToHost;

        // Run the transfer
        cudaMemcpy3D(&p);
        
        cudaFree(bsplineCoeffs.ptr);

        // Replace the pointer
        bsplineCoeffs = temp_ptr;
    }
        
    cudaArray *cuArray = 0;
    // Extent width is in elements for array memory
    cudaExtent ext = make_cudaExtent(width, height, depth);
    // Make a texture out of the array
    CreateTextureFromVolume(&texRef, &cuArray, bsplineCoeffs, ext, enough_gpu);

    // Dispatch to the appropriate free function, depending on where we allocated it
    if (enough_gpu) {
        cudaFree(bsplineCoeffs.ptr);
    } else {
        free(bsplineCoeffs.ptr);
    }
    
    // Get the GPU memory again
    cudaMemGetInfo(&free_mem, &total);

    // Figure out how many slices we can do at once
    printf("free_mem: %zu, total: %zu, %u\n", free_mem, total, new_width*new_height*sizeof(float));
    unsigned int slices = ((free_mem >> 1) + (free_mem >> 2))/(sizeof(float)*new_width*new_height);
    slices = (slices > new_depth) ? new_depth : slices;
    printf("slices: %u %u\n", slices, sizeof(float)*new_width*new_height);
    
    // Allocate the output array on the device
    float *output_dev;
    CUDA_SAFE_CALL(cudaMalloc(&output_dev, sizeof(float)*new_width*new_height*slices));
    // We need this for the cast functions
    cudaPitchedPtr output_pitched = make_cudaPitchedPtr((void*)output_dev, new_width*sizeof(float), new_width, new_height);

    // Allocate the host output array
    T *output_host_running = output_array;

    unsigned int depth_offset = 0;

    // Run until we've done everything
    while (depth_offset < new_depth) {

        // Shrink the number of slices if necessary to not go over
        slices = (depth_offset + slices > new_depth) ? (new_depth - depth_offset) : slices;
        printf("depth_offset %d slices %d\n", depth_offset, slices);

        // Calculate the kernel sizes
        dim3 dimBlock(8, 8, 8);
        dim3 dimGrid((new_width+dimBlock.x-1)/dimBlock.x, (new_height+dimBlock.y-1)/dimBlock.y, (slices+dimBlock.z-1)/dimBlock.z);

        // Run the kernel
        transformKernel<<<dimGrid, dimBlock>>>(output_dev, ((float)width)/(float)new_width, ((float)height)/(float)new_height, ((float)depth)/(float)new_depth, new_width, new_height, slices, depth_offset);
        // Make sure everything's done
        cudaDeviceSynchronize();
        CUT_CHECK_ERROR("kernel failed");

        depth_offset += slices;
        // Copy from the device and cast to the appropriate data type on the host
        CastVolumeDeviceToHost(output_host_running, output_pitched, new_width, new_height, slices);
        output_host_running += slices*new_width*new_height;
    }

    // Free device arrays
    cudaFreeArray(cuArray);
    cudaFree(output_dev);
}

extern "C" void interpolate(
        const float *input_array, float *output_array, unsigned int width, unsigned int height, unsigned int depth,
        unsigned int new_width, unsigned int new_height, unsigned int new_depth
    ) {
    interpolate_template(input_array, output_array, width, height, depth, new_width, new_height, new_depth);
}
