#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
using namespace std;

template <class T>
class MemoryHelper {
 public:
  const vector<int> shape;
  const size_t elem_num = 0;
  const string name;
  const size_t bytes = 0;
  std::unique_ptr<T[]> h_mem = nullptr;
  T* data;

 public:
  MemoryHelper(const vector<int>& shape, T* data_ = nullptr, const string& name = "")
    : shape(shape), name(name), data(data_), elem_num(GetElemNum(shape)), bytes(elem_num * sizeof(T)) {
    if (data_ == nullptr) {
      h_mem = std::make_unique<T[]>(elem_num);
      data = h_mem.get();
    }
  }

  void RandInit(int seed = 0, int max_val = 100, int min_val = 0) {
    srand(seed);
    int gap = max_val - min_val;
    for (size_t i = 0; i < elem_num; i++) {
      data[i] = T((rand() % gap) + min_val);
    }
  }

  void StepInit(float ratio = 0.01f, float bias = 0.0f) {
    for (size_t i = 0; i < elem_num; i++) {
      data[i] = i * ratio + bias;
    }
  }

  T* Mem() { return data; }

  size_t GetBytes() { return bytes; }

  void PrintElems(int sep = 1, size_t max_num = -1, int line_size = 0) {
    printf("print elem of %s begin:\n", name.c_str());
    if (max_num < 0) {
      max_num = elem_num;
    } else {
      max_num = max_num * sep;
    }
    if (line_size <= 0) {
      line_size = elem_num;
    }
    max_num = std::min(elem_num, max_num);
    for (int i = 0; i < max_num; i += sep) {
      std::cout << float(data[i]) << " ";
      if ((i + 1) % line_size == 0) {
        cout << endl;
      }
    }
    printf("\nprint elem of %s end\n", name.c_str());
  }

 public:
  static int GetElemNum(const vector<int>& shape) {
    size_t elem_num = 1;
    for (auto elem : shape) {
      elem_num *= elem;
    }
    return elem_num;
  }
};