#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/hnsw.h>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {
DLManagedTensor tensor(float *data, int64_t *shape)
{
  DLManagedTensor result{};
  result.dl_tensor.data = data;
  result.dl_tensor.device.device_type = kDLCPU;
  result.dl_tensor.device.device_id = 0;
  result.dl_tensor.ndim = 2;
  result.dl_tensor.dtype.code = kDLFloat;
  result.dl_tensor.dtype.bits = 32;
  result.dl_tensor.dtype.lanes = 1;
  result.dl_tensor.shape = shape;
  result.dl_tensor.strides = nullptr;
  result.dl_tensor.byte_offset = 0;
  return result;
}
}

int main()
{
  constexpr int64_t rows = 256;
  constexpr int64_t dim = 8;
  constexpr int64_t topk = 5;
  std::vector<float> dataset(static_cast<size_t>(rows * dim));
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < dim; ++col) {
      dataset[static_cast<size_t>(row * dim + col)] =
          static_cast<float>(row) + 0.01F * static_cast<float>(col);
    }
  }
  int64_t dataset_shape[2] = {rows, dim};
  DLManagedTensor dataset_tensor = tensor(dataset.data(), dataset_shape);

  cuvsResources_t resources = 0;
  cuvsHnswIndexParams_t index_params = nullptr;
  cuvsHnswAceParams_t ace_params = nullptr;
  cuvsHnswIndex_t index = nullptr;
  if (cuvsResourcesCreate(&resources) != CUVS_SUCCESS ||
      cuvsHnswIndexParamsCreate(&index_params) != CUVS_SUCCESS ||
      cuvsHnswAceParamsCreate(&ace_params) != CUVS_SUCCESS ||
      cuvsHnswIndexCreate(&index) != CUVS_SUCCESS) {
    std::cerr << cuvsGetLastErrorText() << std::endl;
    return 1;
  }
  index_params->hierarchy = CPU;
  index_params->metric = L2Expanded;
  index_params->M = 16;
  index_params->ef_construction = 100;
  index_params->ace_params = ace_params;
  if (cuvsHnswBuild(resources, index_params, &dataset_tensor, index) != CUVS_SUCCESS) {
    std::cerr << cuvsGetLastErrorText() << std::endl;
    return 1;
  }

  std::vector<float> query(static_cast<size_t>(dim));
  for (int64_t col = 0; col < dim; ++col) {
    query[static_cast<size_t>(col)] = 42.0F + 0.01F * static_cast<float>(col);
  }
  int64_t query_shape[2] = {1, dim};
  DLManagedTensor query_tensor = tensor(query.data(), query_shape);
  std::vector<uint64_t> neighbors(static_cast<size_t>(topk));
  std::vector<float> distances(static_cast<size_t>(topk));
  int64_t result_shape[2] = {1, topk};
  DLManagedTensor neighbor_tensor = tensor(reinterpret_cast<float *>(neighbors.data()), result_shape);
  neighbor_tensor.dl_tensor.dtype.code = kDLUInt;
  neighbor_tensor.dl_tensor.dtype.bits = 64;
  DLManagedTensor distance_tensor = tensor(distances.data(), result_shape);
  cuvsHnswSearchParams_t search_params = nullptr;
  if (cuvsHnswSearchParamsCreate(&search_params) != CUVS_SUCCESS) {
    std::cerr << cuvsGetLastErrorText() << std::endl;
    return 1;
  }
  search_params->ef = 64;
  if (cuvsHnswSearch(resources, search_params, index,
                    &query_tensor, &neighbor_tensor, &distance_tensor) != CUVS_SUCCESS) {
    std::cerr << cuvsGetLastErrorText() << std::endl;
    return 1;
  }
  std::cout << "nearest=" << neighbors[0] << " distance=" << distances[0] << std::endl;

  cuvsHnswSearchParamsDestroy(search_params);
  cuvsHnswIndexDestroy(index);
  cuvsHnswAceParamsDestroy(ace_params);
  cuvsHnswIndexParamsDestroy(index_params);
  cuvsResourcesDestroy(resources);
  return neighbors[0] == 42 ? 0 : 1;
}
