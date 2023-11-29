#include <chrono>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "mem_helper.h"

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

using TEST_DTYPE = __fp16;
const std::string CL_DTYPE = "half";
auto image_dtype = CL_HALF_FLOAT;

// using TEST_DTYPE = float;
// const std::string CL_DTYPE = "float";
// auto image_dtype = CL_FLOAT;

using namespace std;

std::string ReplaceSubStr(std::string in_str, const std::string &src_rep, const std::string &dst_rep) {
  if (!src_rep.empty()) {
    in_str = std::regex_replace(in_str, std::regex(src_rep), dst_rep);
  }
  return in_str;
}

/*
ref
https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-1-opencl-optimization
https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
https://github.com/ysh329/OpenCL-101/issues/55
*/

const std::string gemm_kernel{R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define DTYPE       DTYPE_ARG
#define DTYPE_PACK4 DTYPE_ARG4

#define READ_IMAGE_FUNC  READ_IMAGE_FUNC_ARG
#define WRITE_IMAGE_FUNC WRITE_IMAGE_FUNC_ARG

#define KTILE 4

__kernel void gemm_kernel(__read_only image2d_t img_a,
                          __read_only image2d_t img_b,
                          __write_only image2d_t img_c,
                          const int m, const int n, const int k) {
  int gx = get_global_id(0); // [0, N / 4)
  int gy = get_global_id(1); // [0, M / 8)

  DTYPE_PACK4 a[8];
  DTYPE_PACK4 b[4];
  DTYPE_PACK4 c[8];

  for (int i = 0; i < 8; i++) {
    c[i] = 0.0f;
  }

  int a_y_off = gy << 3;

  int kloop = k / KTILE;

  for (int ktcnt = 0; ktcnt < kloop; ktcnt++) {
    int kpos = ktcnt*KTILE;
    
// 准备数据b：1行4列
#pragma unroll
    for (int i = 0; i < 4; i++) {
      b[i] = READ_IMAGE_FUNC(img_b, (int2)(gx, kpos + i));
    }

    // 准备数据a: 8行4列（4列为float4）
#pragma unroll
    for (int i = 0; i < 8; i++) {
      a[i] = READ_IMAGE_FUNC(img_a, (int2)(ktcnt, a_y_off + i));
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
      c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];
    }
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int c_offs = (gy << 3) + i;

    WRITE_IMAGE_FUNC(img_c, (int2)(gx, c_offs), c[i]);
  }
}
)"};

int main() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  std::cout << "get platform num:" << platforms.size() << std::endl;

  cl::Platform plat;
  for (auto &p : platforms) {
    std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
    if (platver.find("OpenCL 2.") != std::string::npos || platver.find("OpenCL 3.") != std::string::npos) {
      // Note: an OpenCL 3.x platform may not support all required features!
      plat = p;
    }
  }
  if (plat() == 0) {
    std::cout << "No OpenCL 2.0 or newer platform found.\n";
    return -1;
  }

  std::cout << "platform name:" << plat.getInfo<CL_PLATFORM_NAME>() << std::endl;

  cl::Platform newP = cl::Platform::setDefault(plat);
  if (newP != plat) {
    std::cout << "Error setting default platform.\n";
    return -1;
  }

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  newP.getDevices(CL_DEVICE_TYPE_GPU, &all_devices); // CL_DEVICE_TYPE_ALL
  std::cout << "get all_devices num:" << all_devices.size() << std::endl;

  if (all_devices.size() == 0) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    exit(1);
  }
  cl_int error;

  // cl::Device default_device = cl::Device::getDefault();
  cl::Device default_device = all_devices[0];
  std::cout << "device name: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

  // a context is like a "runtime link" to the device and platform;
  // i.e. communication is possible
  cl::Context context({default_device});
  int queue_properties = 0;
  queue_properties |= CL_QUEUE_PROFILING_ENABLE;
  cl::CommandQueue queue(context, default_device, queue_properties);

  // m must be integer multiple of 8
  // n, k must be integer multiple of 4
  int m = 1024;
  int n = 1024;
  int k = 1024;

  vector<int> a_shape = {m, k};
  vector<int> b_shape = {k, n};
  vector<int> c_shape = {m, n};

  MemoryHelper<TEST_DTYPE> mem_a(a_shape);
  MemoryHelper<TEST_DTYPE> mem_b(b_shape);
  MemoryHelper<TEST_DTYPE> mem_c(c_shape);
  mem_a.StepInit(0.0f, 0.1f);
  mem_b.StepInit(0.0f, 0.1f);
  memset(mem_c.Mem(), 0, mem_c.bytes);

  // CL_MEM_WRITE_ONLY CL_MEM_READ_ONLY CL_MEM_READ_WRITE
  cl::Buffer d_a = cl::Buffer(context, CL_MEM_READ_WRITE, mem_a.bytes);
  // cl::Buffer d_b = cl::Buffer(context, CL_MEM_READ_WRITE, mem_b.bytes);
  cl::Buffer d_c = cl::Buffer(context, CL_MEM_READ_WRITE, mem_c.bytes);

  cl_image_format image_format;
  image_format.image_channel_order = CL_RGBA;
  image_format.image_channel_data_type = image_dtype;

  cl_image_desc image_desc_a = {0};
  image_desc_a.image_type = CL_MEM_OBJECT_IMAGE2D;
  image_desc_a.image_width = k / 4;
  image_desc_a.image_height = m;
  image_desc_a.image_row_pitch = 0;

  cl_image_desc image_desc_b = {0};
  image_desc_b.image_type = CL_MEM_OBJECT_IMAGE2D;
  image_desc_b.image_width = n / 4;
  image_desc_b.image_height = k;
  image_desc_b.image_row_pitch = 0;

  cl_image_desc image_desc_c = {0};
  image_desc_c.image_type = CL_MEM_OBJECT_IMAGE2D;
  image_desc_c.image_width = n / 4;
  image_desc_c.image_height = m;
  image_desc_c.image_row_pitch = 0;

  cl_mem img_a = clCreateImage(context.get(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc_a,
                               NULL, &error);

  cl_mem img_b = clCreateImage(context.get(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc_b,
                               NULL, &error);

  cl_mem img_c = clCreateImage(context.get(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, &image_format, &image_desc_c,
                               NULL, &error);

  array<size_t, 3> origin;
  array<size_t, 3> region;

  origin[0] = 0;
  origin[1] = 0;
  origin[2] = 0;

  region[0] = k / 4;
  region[1] = m;
  region[2] = 1;

  error |=
      clEnqueueWriteImage(queue.get(), img_a, CL_TRUE, origin.data(), region.data(), 0, 0, mem_a.Mem(), 0, NULL, NULL);

  region[0] = n / 4;
  region[1] = k;
  region[2] = 1;

  error |=
      clEnqueueWriteImage(queue.get(), img_b, CL_TRUE, origin.data(), region.data(), 0, 0, mem_b.Mem(), 0, NULL, NULL);

  string kernel_code = gemm_kernel;
  kernel_code = ReplaceSubStr(kernel_code, "DTYPE_ARG", CL_DTYPE);
  if (CL_DTYPE == "float") {
    kernel_code = ReplaceSubStr(kernel_code, "READ_IMAGE_FUNC_ARG", "read_imagef");
    kernel_code = ReplaceSubStr(kernel_code, "WRITE_IMAGE_FUNC_ARG", "write_imagef");
  } else {
    kernel_code = ReplaceSubStr(kernel_code, "READ_IMAGE_FUNC_ARG", "read_imageh");
    kernel_code = ReplaceSubStr(kernel_code, "WRITE_IMAGE_FUNC_ARG", "write_imageh");
  }

  std::vector<std::string> programStrings;
  programStrings.push_back(kernel_code);
  cl::Program program(context, programStrings);

  if (program.build({default_device}, "-cl-std=CL3.0") != CL_SUCCESS) {
    std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
    exit(1);
  }

  cl::Kernel cl_kernel(program, "gemm_kernel");

  error = 0;
  int arg_pos = 0;
  error |= cl_kernel.setArg(arg_pos++, sizeof(cl_mem), &img_a);

  error |= cl_kernel.setArg(arg_pos++, sizeof(cl_mem), &img_b);
  if (error != CL_SUCCESS) {
    printf("setArg1 failed\n");
  }

  error |= cl_kernel.setArg(arg_pos++, sizeof(cl_mem), &img_c);
  if (error != CL_SUCCESS) {
    printf("setArg2 failed\n");
  }

  error |= cl_kernel.setArg(arg_pos++, sizeof(int), &m);
  error |= cl_kernel.setArg(arg_pos++, sizeof(int), &n);
  error |= cl_kernel.setArg(arg_pos++, sizeof(int), &k);

  if (error != CL_SUCCESS) {
    printf("setArg failed\n");
  }

  int total_threads_x = n / 4;
  int total_threads_y = m / 8;
  int local_threads_x = 16;
  int local_threads_y = 16;

  local_threads_x = std::min(local_threads_x, total_threads_x);
  local_threads_y = std::min(local_threads_y, total_threads_y);

  cout << "local_threads_x:" << local_threads_x << endl;
  cout << "local_threads_y:" << local_threads_y << endl;
  cout << "total_threads_x:" << total_threads_x << endl;
  cout << "total_threads_y:" << total_threads_y << endl;

  cl::NDRange global_size(total_threads_x, total_threads_y);
  cl::NDRange local_size(local_threads_x, local_threads_y);

  int eval_num = 4;
  for (int i = 0; i < eval_num; i++) {

    cl::Event event;
    cl_int err = queue.enqueueNDRangeKernel(cl_kernel, cl::NullRange, global_size, local_size, NULL, &event);
    if (err != CL_SUCCESS) {
      printf("enqueueNDRangeKernel failed\n");
    }

    event.wait();
    cl_ulong start_time, end_time; // time in ns
    cl_int err1 = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    cl_int err2 = event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
    float exec_time = (end_time - start_time) / 1000.0f;
    printf("mean exec time: %f us ----------\n", exec_time);
  }
  queue.finish();

  region[0] = n / 4;
  region[1] = m;
  region[2] = 1;

  error |=
      clEnqueueReadImage(queue.get(), img_c, CL_TRUE, origin.data(), region.data(), 0, 0, mem_c.Mem(), 0, NULL, NULL);

  std::cout << "results:" << std::endl;
  mem_a.PrintElems(1, 128, 64);
  mem_b.PrintElems(1, 128, 64);
  mem_c.PrintElems(1, 128, 64);
  return 0;
}