#include <cuvs/neighbors/hnsw.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <iostream>

int main()
{
  raft::resources resources;
  constexpr int64_t rows = 256;
  constexpr int64_t dim = 8;
  constexpr int64_t topk = 5;
  auto dataset = raft::make_host_matrix<float, int64_t>(resources, rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < dim; ++col) {
      dataset(row, col) = static_cast<float>(row) + 0.01F * static_cast<float>(col);
    }
  }

  cuvs::neighbors::hnsw::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  index_params.hierarchy = cuvs::neighbors::hnsw::HnswHierarchy::CPU;
  index_params.M = 16;
  index_params.ef_construction = 100;
  auto index = cuvs::neighbors::hnsw::build(
      resources, index_params, raft::make_const_mdspan(dataset.view()));

  auto queries = raft::make_host_matrix<float, int64_t>(resources, 1, dim);
  for (int64_t col = 0; col < dim; ++col) {
    queries(0, col) = 42.0F + 0.01F * static_cast<float>(col);
  }
  auto neighbors = raft::make_host_matrix<uint64_t, int64_t>(resources, 1, topk);
  auto distances = raft::make_host_matrix<float, int64_t>(resources, 1, topk);
  cuvs::neighbors::hnsw::search_params search_params;
  search_params.ef = 64;
  cuvs::neighbors::hnsw::search(
      resources, search_params, *index, queries.view(), neighbors.view(), distances.view());

  std::cout << "nearest=" << neighbors(0, 0)
            << " distance=" << distances(0, 0) << std::endl;
  return neighbors(0, 0) == 42 ? 0 : 1;
}
