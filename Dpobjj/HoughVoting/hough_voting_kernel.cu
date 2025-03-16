#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cfloat>
#include <time.h>
#include <thrust/extrema.h>
#include <Eigen/Geometry>
#include <cublas_v2.h>
#include <torch/extension.h>

#define VERTEX_CHANNELS 3
#define MAX_ROI 128

// Macro to iterate over CUDA threads in a 1D kernel.
#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);    \
       i += blockDim.x * gridDim.x)

//---------------------------------------------------------------
// Helper Device Functions
//---------------------------------------------------------------

// Computes the Euclidean norm of a 2D vector.
__device__ inline float norm2d(float a, float b) {
    return sqrtf(a * a + b * b);
}

// Computes the cosine similarity (angle distance) between the vector (u, v)
// and the vector from (x, y) to (cx, cy).
// A small epsilon (1e-6f) is added to the denominator to avoid division by zero.
__device__ inline float angle_distance(int cx, int cy, int x, int y, float u, float v) {
    float dx = cx - x;
    float dy = cy - y;
    float n1 = norm2d(u, v);
    float n2 = norm2d(dx, dy);
    float dot = u * dx + v * dy;
    return dot / (n1 * n2 + 1e-6f);
}

// Computes the angle distance as above, but also samples points along the line
// from (x, y) to (cx, cy) to ensure that a sufficient fraction belong to class 'cls'.
// If less than 50% of the sampled points have the same label, the distance is set to 0.
__device__ inline float angle_distance_label(int cx, int cy, int x, int y, float u, float v,
                                              int cls, const int height, const int width,
                                              const int* labelmap) {
    float dx = cx - x;
    float dy = cy - y;
    float n1 = norm2d(u, v);
    float n2 = norm2d(dx, dy);
    float dot = u * dx + v * dy;
    float distance = dot / (n1 * n2 + 1e-6f);

    const int num_steps = 10;
    int count = 0;
    for (int i = 1; i <= num_steps; i++) {
        float step = float(i) / float(num_steps);
        int px = int(x + step * dx);
        int py = int(y + step * dy);
        if (px >= 0 && px < width && py >= 0 && py < height) {
            if (labelmap[py * width + px] == cls)
                count++;
        }
    }
    if (static_cast<float>(count) / num_steps < 0.5f)
        distance = 0.0f;

    return distance;
}

// Projects a 3D bounding box based on the given extents and meta data.
// The function computes the 8 vertices of the 3D box, projects them into 2D
// using camera intrinsics (fx, fy, px, py), then computes the bounding box
// in the image. The maximum dimension of this box, scaled by 'factor', is
// returned in 'threshold'.
__device__ inline void project_box(int cls, const float* extents, const float* meta_data,
                                   float distance, float factor, float* threshold) {
    float xHalf = extents[cls * 3 + 0] * 0.5f;
    float yHalf = extents[cls * 3 + 1] * 0.5f;
    float zHalf = extents[cls * 3 + 2] * 0.5f;

    // Compute the 3D bounding box vertices (8 vertices, each with 3 coordinates).
    float bb3D[24] = {
         xHalf,  yHalf,  zHalf + distance,
        -xHalf,  yHalf,  zHalf + distance,
         xHalf, -yHalf,  zHalf + distance,
        -xHalf, -yHalf,  zHalf + distance,
         xHalf,  yHalf, -zHalf + distance,
        -xHalf,  yHalf, -zHalf + distance,
         xHalf, -yHalf, -zHalf + distance,
        -xHalf, -yHalf, -zHalf + distance
    };

    // Retrieve intrinsic camera parameters.
    float fx = meta_data[0];
    float fy = meta_data[4];
    float px = meta_data[2];
    float py = meta_data[5];

    float minX = 1e8f, minY = 1e8f;
    float maxX = -1e8f, maxY = -1e8f;
    
    // Project the 3D vertices to 2D image coordinates.
    for (int i = 0; i < 8; i++) {
        float invZ = 1.0f / bb3D[i * 3 + 2];
        float x_proj = fx * (bb3D[i * 3] * invZ) + px;
        float y_proj = fy * (bb3D[i * 3 + 1] * invZ) + py;
        minX = fminf(minX, x_proj);
        minY = fminf(minY, y_proj);
        maxX = fmaxf(maxX, x_proj);
        maxY = fmaxf(maxY, y_proj);
    }
    
    float width_box = maxX - minX + 1;
    float height_box = maxY - minY + 1;
    *threshold = fmaxf(width_box, height_box) * factor;
}

//---------------------------------------------------------------
// CUDA Kernel: compute_arrays_kernel
//---------------------------------------------------------------
// This kernel processes the 'labelmap' (an image of class labels) and, for
// each non-zero label, atomically adds the index of that pixel to a per-class array.
// 'arrays' holds the indices per class, and 'array_size' tracks the number of indices per class.
__global__ void compute_arrays_kernel(const int nthreads, const int* labelmap,
                                        int* arrays, int* array_size,
                                        const int height, const int width) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int cls = labelmap[index];
        if (cls > 0) {
            int size = atomicAdd(array_size + cls, 1);
            int offset = cls * height * width + size;
            arrays[offset] = index;
        }
    }
}

__global__ void compute_hough_kernel(const int nthreads, float* hough_space, float* hough_data, 
                                     const int* labelmap, const float* vertmap, const float* extents, 
                                     const float* meta_data, int* arrays, int* array_size, 
                                     int* class_indexes, const int height, const int width, 
                                     const int num_classes, const int count, 
                                     const float inlierThreshold, const int skip_pixels) 
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) 
    {
        // Compute the class index and (cx, cy) in the Hough space
        int ind = index / (height * width);
        int cls = class_indexes[ind];
        int n = index % (height * width);
        int cx = n % width;
        int cy = n / width;

        int size = array_size[cls];
        float accumulated_distance = 0.0f;
        float threshold;

        // Voting process
        for (int i = 0; i < size; i += skip_pixels)
        {
            int offset = cls * height * width + i;
            int location = arrays[offset];
            int x = location % width;
            int y = location / width;

            // Read the vertex direction
            int vert_offset = VERTEX_CHANNELS * cls * height * width + y * width + x;
            float u = vertmap[vert_offset];
            float v = vertmap[vert_offset + height * width];
            float d = expf(vertmap[vert_offset + 2 * height * width]);

            // Check if the angle distance is within the inlier threshold
            if (angle_distance_label(cx, cy, x, y, u, v, cls, height, width, labelmap) > inlierThreshold)
            {
                project_box(cls, extents, meta_data, d, 0.6f, &threshold);
                float dx = fabsf(x - cx);
                float dy = fabsf(y - cy);
                if (dx < threshold && dy < threshold)
                {
                    hough_space[index]++;
                    accumulated_distance += d;
                }
            }
        }

        // Compute the final distance value
        if (hough_space[index] > 0)
        {
            accumulated_distance /= hough_space[index];

            // Compute bounding box dimensions
            float bb_width = -1.0f, bb_height = -1.0f;
            for (int i = 0; i < size; i += skip_pixels)
            {
                int offset = cls * height * width + i;
                int location = arrays[offset];
                int x = location % width;
                int y = location / width;

                // Read the vertex direction
                int vert_offset = VERTEX_CHANNELS * cls * height * width + y * width + x;
                float u = vertmap[vert_offset];
                float v = vertmap[vert_offset + height * width];

                // Check if the angle distance is within the inlier threshold
                if (angle_distance_label(cx, cy, x, y, u, v, cls, height, width, labelmap) > inlierThreshold)
                {
                    project_box(cls, extents, meta_data, accumulated_distance, 0.6f, &threshold);
                    float dx = fabsf(x - cx);
                    float dy = fabsf(y - cy);
                    if (dx > bb_width && dx < threshold && dy < threshold) bb_width = dx;
                    if (dy > bb_height && dx < threshold && dy < threshold) bb_height = dy;
                }
            }

            // Store results in hough_data
            int data_offset = ind * height * width * 3 + 3 * (cy * width + cx);
            hough_data[data_offset] = accumulated_distance;
            hough_data[data_offset + 1] = 2.0f * bb_height;
            hough_data[data_offset + 2] = 2.0f * bb_width;
        }
    }
}

__global__ void compute_max_indexes_kernel(const int nthreads, int* max_indexes, int index_size, 
                                           int* num_max, float* hough_space, float* hough_data, 
                                           int height, int width, float threshold, 
                                           float perThreshold, const int is_train) 
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) 
    {
        // Compute (ind, cx, cy) in the Hough space
        int ind = index / (height * width);
        int n = index % (height * width);
        int cx = n % width;
        int cy = n / width;

        // Read bounding box dimensions
        int offset = ind * height * width * 3 + 3 * (cy * width + cx);
        float bb_height = hough_data[offset + 1];
        float bb_width = hough_data[offset + 2];

        float vote_count = hough_space[index];

        // Ensure threshold conditions are met
        if (vote_count <= threshold || bb_height <= 0 || bb_width <= 0)
            return;

        // Check for local maximum within a 3x3 neighborhood
        int kernel_size = 3;
        bool is_local_max = true;
        int max_offset = ind * height * width; // Avoid redundant computation

        for (int x = cx - kernel_size; x <= cx + kernel_size && is_local_max; x++) 
        {
            for (int y = cy - kernel_size; y <= cy + kernel_size; y++) 
            {
                if (x >= 0 && x < width && y >= 0 && y < height) 
                {
                    int neighbor_index = max_offset + y * width + x;
                    float neighbor_vote = hough_space[neighbor_index];

                    if (neighbor_vote > vote_count || 
                        (is_train == 0 && neighbor_vote == vote_count && neighbor_index > index)) 
                    {
                        is_local_max = false;
                        break; // Early exit
                    }
                }
            }
        }

        // Check voting percentage condition
        if (vote_count / (bb_height * bb_width) < perThreshold)
            is_local_max = false;

        // Store the local maximum index if conditions are met
        if (is_local_max) 
        {
            int max_index = atomicAdd(num_max, 1);
            if (max_index < index_size)
                max_indexes[max_index] = index;
        }
    }
}

__global__ void compute_rois_kernel(const int nthreads, float* top_box, float* top_pose, 
    const float* meta_data, float* hough_space, float* hough_data, int* max_indexes, int* class_indexes,
    int batch_index, const int height, const int width, const int num_classes, int* num_rois, const int is_train) 
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) 
    {
        int max_index = max_indexes[index];
        int ind = max_index / (height * width);
        int cls = class_indexes[ind];
        int n = max_index % (height * width);
        int x = n % width;
        int y = n / width;

        float fx = meta_data[0];
        float fy = meta_data[4];
        float px = meta_data[2];
        float py = meta_data[5];
        float rx = (x - px) / fx;
        float ry = (y - py) / fy;

        int offset = ind * height * width * 3 + 3 * (y * width + x);
        float bb_distance = hough_data[offset];
        float bb_height = hough_data[offset + 1];
        float bb_width = hough_data[offset + 2];

        float hough_val = hough_space[max_index];
        float scale = 0.0;

        // Atomic update for ROI index
        int base_roi_index = atomicAdd(num_rois, is_train ? 9 : 1);
        
        // Store primary ROI
        int roi_offset = base_roi_index * 7;
        top_box[roi_offset + 0] = batch_index;
        top_box[roi_offset + 1] = cls;
        top_box[roi_offset + 2] = x - bb_width * (0.5 + scale);
        top_box[roi_offset + 3] = y - bb_height * (0.5 + scale);
        top_box[roi_offset + 4] = x + bb_width * (0.5 + scale);
        top_box[roi_offset + 5] = y + bb_height * (0.5 + scale);
        top_box[roi_offset + 6] = hough_val;

        if (is_train) 
        {
            // Define offsets for jittering boxes
            float jitter_offsets[8][2] = {
                {-0.05, -0.05}, {+0.05, -0.05}, {-0.05, +0.05}, {+0.05, +0.05},
                { 0.00, -0.05}, {-0.05,  0.00}, { 0.00, +0.05}, {+0.05,  0.00}
            };

            float x1 = top_box[roi_offset + 2];
            float y1 = top_box[roi_offset + 3];
            float x2 = top_box[roi_offset + 4];
            float y2 = top_box[roi_offset + 5];
            float ww = x2 - x1;
            float hh = y2 - y1;

            for (int j = 0; j < 8; j++) 
            {
                int jitter_roi_offset = (base_roi_index + j + 1) * 7;
                top_box[jitter_roi_offset + 0] = batch_index;
                top_box[jitter_roi_offset + 1] = cls;
                top_box[jitter_roi_offset + 2] = x1 + jitter_offsets[j][0] * ww;
                top_box[jitter_roi_offset + 3] = y1 + jitter_offsets[j][1] * hh;
                top_box[jitter_roi_offset + 4] = top_box[jitter_roi_offset + 2] + ww;
                top_box[jitter_roi_offset + 5] = top_box[jitter_roi_offset + 3] + hh;
                top_box[jitter_roi_offset + 6] = hough_val;
            }

            // Store pose information for all ROIs
            for (int j = 0; j < 9; j++) 
            {
                int pose_offset = (base_roi_index + j) * 7;
                top_pose[pose_offset + 0] = 1;
                top_pose[pose_offset + 1] = 0;
                top_pose[pose_offset + 2] = 0;
                top_pose[pose_offset + 3] = 0;
                top_pose[pose_offset + 4] = rx;
                top_pose[pose_offset + 5] = ry;
                top_pose[pose_offset + 6] = bb_distance;
            }
        }
        else 
        {
            // Store pose information for a single ROI
            int pose_offset = base_roi_index * 7;
            top_pose[pose_offset + 0] = 1;
            top_pose[pose_offset + 1] = 0;
            top_pose[pose_offset + 2] = 0;
            top_pose[pose_offset + 3] = 0;
            top_pose[pose_offset + 4] = rx;
            top_pose[pose_offset + 5] = ry;
            top_pose[pose_offset + 6] = bb_distance;
        }
    }
}

#include <vector>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

std::vector<at::Tensor> hough_voting_cuda_forward(
    at::Tensor bottom_label,
    at::Tensor bottom_vertex,
    at::Tensor bottom_meta_data,
    at::Tensor extents,
    int is_train,
    int skip_pixels,
    int labelThreshold,
    float inlierThreshold,
    float votingThreshold,
    float perThreshold)
{
    // CUDA Debugging Start
    printf("Entering hough_voting_cuda_forward...\n");
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA Error at start: %s\n", cudaGetErrorString(err));
    }

    const int kThreadsPerBlock = 512;
    const int batch_size = bottom_vertex.size(0);
    const int num_classes = bottom_vertex.size(1) / VERTEX_CHANNELS;
    const int height = bottom_vertex.size(2);
    const int width = bottom_vertex.size(3);
    const int num_meta_data = bottom_meta_data.size(1);
    const int index_size = MAX_ROI / batch_size;

    printf("Batch Size: %d, Num Classes: %d, Height: %d, Width: %d\n", batch_size, num_classes, height, width);

    auto top_box = at::zeros({MAX_ROI * 9, 7}, bottom_vertex.options());
    auto top_pose = at::zeros({MAX_ROI * 9, 7}, bottom_vertex.options());
    auto num_rois = at::zeros({1}, bottom_label.options());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int batch_index = 0; batch_index < batch_size; batch_index++)
    {
        printf("Processing Batch %d...\n", batch_index);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] Before kernel launch: %s\n", cudaGetErrorString(err));
        }

        const int* labelmap = bottom_label.data_ptr<int>() + batch_index * height * width;
        const float* vertmap = bottom_vertex.data_ptr<float>() + batch_index * height * width * VERTEX_CHANNELS * num_classes;
        const float* meta_data = bottom_meta_data.data_ptr<float>() + batch_index * num_meta_data;

        auto arrays = at::zeros({num_classes, height * width}, bottom_label.options());
        auto array_sizes = at::zeros({num_classes}, bottom_label.options());

        printf("Launching compute_arrays_kernel...\n");
        compute_arrays_kernel<<<(height * width + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            height * width, labelmap, arrays.data_ptr<int>(), array_sizes.data_ptr<int>(), height, width);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] compute_arrays_kernel failed: %s\n", cudaGetErrorString(err));
        }

        std::vector<int> array_sizes_host(num_classes);
        cudaMemcpyAsync(array_sizes_host.data(), array_sizes.data_ptr<int>(), num_classes * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        printf("Computing class indexes...\n");
        std::vector<int> class_indexes_host;
        for (int c = 1; c < num_classes; c++) {
            if (array_sizes_host[c] > labelThreshold) {
                class_indexes_host.push_back(c);
            }
        }
        if (class_indexes_host.empty()) {
            printf("Skipping batch %d (no valid classes)\n", batch_index);
            continue;
        }

        auto class_indexes = at::tensor(class_indexes_host, bottom_label.options());
        auto hough_space = at::zeros({(int)class_indexes_host.size(), height, width}, bottom_vertex.options());
        auto hough_data = at::zeros({(int)class_indexes_host.size(), height, width, 3}, bottom_vertex.options());

        printf("Launching compute_hough_kernel...\n");
        compute_hough_kernel<<<((int)class_indexes_host.size() * height * width + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            (int)class_indexes_host.size() * height * width, hough_space.data_ptr<float>(), hough_data.data_ptr<float>(),
            labelmap, vertmap, extents.data_ptr<float>(), meta_data, arrays.data_ptr<int>(), array_sizes.data_ptr<int>(),
            class_indexes.data_ptr<int>(), height, width, num_classes, (int)class_indexes_host.size(), inlierThreshold, skip_pixels);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] compute_hough_kernel failed: %s\n", cudaGetErrorString(err));
        }

        auto num_max = at::zeros({1}, bottom_label.options());
        auto max_indexes = at::zeros({index_size}, bottom_label.options());

        printf("Determining max indexes...\n");
        if (votingThreshold > 0) {
            compute_max_indexes_kernel<<<((int)class_indexes_host.size() * height * width + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
                (int)class_indexes_host.size() * height * width, max_indexes.data_ptr<int>(), index_size, num_max.data_ptr<int>(),
                hough_space.data_ptr<float>(), hough_data.data_ptr<float>(), height, width, votingThreshold, perThreshold, is_train);
        } else {
            thrust::device_ptr<float> d_hough_space(hough_space.data_ptr<float>());
            thrust::device_vector<int> max_indexes_vec((int)class_indexes_host.size());
            for (int i = 0; i < (int)class_indexes_host.size(); i++) {
                auto hmax = thrust::max_element(thrust::device, d_hough_space + i * height * width, d_hough_space + (i + 1) * height * width);
                max_indexes_vec[i] = hmax - d_hough_space;
            }
            cudaMemcpyAsync(max_indexes.data_ptr<int>(), thrust::raw_pointer_cast(max_indexes_vec.data()), class_indexes_host.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
            int class_count = class_indexes_host.size();
            cudaMemcpyAsync(num_max.data_ptr<int>(), &class_count, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        cudaDeviceSynchronize();

        int num_max_host;
        cudaMemcpy(&num_max_host, num_max.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
        num_max_host = std::min(num_max_host, index_size);

        if (num_max_host > 0) {
            printf("Launching compute_rois_kernel...\n");
            compute_rois_kernel<<<(num_max_host + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
                num_max_host, top_box.data_ptr<float>(), top_pose.data_ptr<float>(), meta_data,
                hough_space.data_ptr<float>(), hough_data.data_ptr<float>(), max_indexes.data_ptr<int>(),
                class_indexes.data_ptr<int>(), batch_index, height, width, num_classes, num_rois.data_ptr<int>(), is_train);
        }
    }

    cudaStreamDestroy(stream);
    cudaDeviceSynchronize();

    printf("Finalizing output tensors...\n");
    int num_rois_host;
    cudaMemcpy(&num_rois_host, num_rois.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    num_rois_host = std::max(num_rois_host, 1);
    auto top_box_final = at::zeros({num_rois_host, 7}, bottom_vertex.options());
    auto top_pose_final = at::zeros({num_rois_host, 7}, bottom_vertex.options());
    cudaMemcpy(top_box_final.data_ptr<float>(), top_box.data_ptr<float>(), num_rois_host * 7 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(top_pose_final.data_ptr<float>(), top_pose.data_ptr<float>(), num_rois_host * 7 * sizeof(float), cudaMemcpyDeviceToDevice);

    printf("Exiting hough_voting_cuda_forward\n");
    return {top_box_final, top_pose_final};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hough_voting_cuda_forward", &hough_voting_cuda_forward, "Hough Voting Forward CUDA");
}