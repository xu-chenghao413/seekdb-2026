/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX LIB

#include "ob_vsag_adaptor.h"
#include <map>
#include "vsag/vsag.h"
#include "vsag/errors.h"
#include "vsag/dataset.h"
#include "vsag/search_param.h"
#include "vsag/index.h"
#include "vsag/options.h"
#include "vsag/factory.h"
#include "lib/utility/ob_print_utils.h"
#include "lib/oblog/ob_log.h"
#include "lib/worker.h"

#ifdef OB_ENABLE_CUVS
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/hnsw.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <strings.h>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#endif

namespace oceanbase {
namespace common {
namespace obvsag {

using namespace vsag;

static int vsag_errcode2ob(vsag::ErrorType vsag_errcode)
{
  int ret = OB_ERR_VSAG_RETURN_ERROR;
  switch (vsag_errcode) {
    case vsag::ErrorType::INVALID_ARGUMENT: {
      ret = OB_INVALID_ARGUMENT;
      LOG_WARN("invalid vsag parameter", K(ret), K(vsag_errcode));
      break;
    }
    case vsag::ErrorType::UNSUPPORTED_INDEX:
    case vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION: {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("not support vsag feature", K(ret), K(vsag_errcode));
      break;
    }
    case vsag::ErrorType::DIMENSION_NOT_EQUAL: {
      ret = OB_INVALID_ARGUMENT;
      LOG_WARN("the dimension of request is NOT equal to index", K(ret), K(vsag_errcode));
      break;
    }
    case vsag::ErrorType::INDEX_EMPTY: {
      ret = OB_OP_NOT_ALLOW;
      LOG_WARN("index is empty, cannot search or serialize", K(ret), K(vsag_errcode));
      break;
    }
    case vsag::ErrorType::NO_ENOUGH_MEMORY: {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("failed to alloc memory in vasg", K(ret), K(vsag_errcode));
      break;
    }
    default: {
      ret = OB_ERR_VSAG_RETURN_ERROR;
      LOG_WARN("vsag return error", K(ret), K(vsag_errcode));
      break;
    }
  }
  return ret;
}

static void fill_vsag_error_message(const vsag::Error &error, std::string &err_msg)
{
  err_msg = error.message;
}

static void adjust_create_index_max_degree(const IndexType index_type, int &max_degree)
{
  // hgraph of vsag needs to be multiplied by 2 so as to align recall with hnsw
  if (HNSW_SQ_TYPE == index_type || HNSW_BQ_TYPE == index_type || HGRAPH_TYPE == index_type) {
    max_degree *= 2;
    LOG_INFO("change max_degree for hgraph", K(index_type), K(max_degree), K(lbt()));
  }
}

class ObVasgFilter final : public vsag::Filter {
public:
  ObVasgFilter(float valid_ratio,
               const std::function<bool(int64_t)> &vid_fallback_func,
               const std::function<bool(const char *)> &exinfo_fallback_func)
      : valid_ratio_(valid_ratio), vid_fallback_func_(vid_fallback_func),
        exinfo_fallback_func_(exinfo_fallback_func){};

  ~ObVasgFilter() {}

  bool CheckValid(int64_t id) const override { return !vid_fallback_func_(id); }

  bool CheckValid(const char *data) const override {
    return !exinfo_fallback_func_(data);
  }

  float ValidRatio() const override { return valid_ratio_; }

private:
  float valid_ratio_;
  std::function<bool(int64_t)> vid_fallback_func_{nullptr};
  std::function<bool(const char *)> exinfo_fallback_func_{nullptr};
};

class HnswIndexHandler {
public:
  HnswIndexHandler(bool is_create, bool is_build, bool use_static,
                   const char *dtype, const char *metric, int max_degree,
                   int ef_construction, int ef_search, int dim,
                   IndexType index_type, std::shared_ptr<vsag::Index> index,
                   vsag::Allocator *allocator, uint64_t extra_info_size,
                   int16_t refine_type, int16_t bq_bits_query, bool bq_use_fht)
      : magic_(VSAG_HANDLER_MAGIC), is_created_(is_create), is_build_(is_build), use_static_(use_static),
        dtype_(dtype), metric_(metric), max_degree_(max_degree),
        ef_construction_(ef_construction), ef_search_(ef_search), dim_(dim),
        index_type_(index_type), index_(index), allocator_(allocator),
        extra_info_size_(extra_info_size), refine_type_(refine_type),
        bq_bits_query_(bq_bits_query), bq_use_fht_(bq_use_fht) {}

  HnswIndexHandler(bool is_create, bool is_build, bool use_static, const char *dtype, const char *metric,
      IndexType index_type, std::shared_ptr<vsag::Index> index, vsag::Allocator *allocator, uint64_t extra_info_size,
      bool use_reorder, float doc_prune_ratio, int window_size)
      : magic_(VSAG_HANDLER_MAGIC), is_created_(is_create),
        is_build_(is_build),
        use_static_(use_static),
        dtype_(dtype),
        metric_(metric),
        index_type_(index_type),
        index_(index),
        allocator_(allocator),
        extra_info_size_(extra_info_size),
        use_reorder_(use_reorder),
        doc_prune_ratio_(doc_prune_ratio),
        window_size_(window_size)
  {}

  ~HnswIndexHandler() {
    index_ = nullptr;
    LOG_INFO("[OBVSAG] after deconstruction, hnsw index", KP(allocator_), K(index_.use_count()), K(lbt()));
  }
  int build_index(const vsag::DatasetPtr &base);
  int get_index_number();
  int add_index(const vsag::DatasetPtr &incremental);
  int cal_distance_by_id(const float *vector, const int64_t *ids, int64_t count,
                         const float *&dist);
  int cal_distance_by_id(uint32_t len, uint32_t *dims, float *vals,
                        const int64_t *ids, int64_t count, const float *&dist);
  int get_extra_info_by_ids(const int64_t *ids, int64_t count,
                            char *extra_infos);
  int get_vid_bound(int64_t &min_vid, int64_t &max_vid);
  uint64_t estimate_memory(const uint64_t row_count, const bool is_build);
  int knn_search(const vsag::DatasetPtr &query, int64_t topk,
                 const std::string &parameters, const float *&dist,
                 const int64_t *&ids, int64_t &result_size, float valid_ratio,
                 int index_type, FilterInterface *bitmap, bool reverse_filter,
                 bool need_extra_info, const char *&extra_infos,
                 void *allocator, float distance_threshold = FLT_MAX);
  int knn_search(const vsag::DatasetPtr &query, int64_t topk,
                 const std::string &parameters, const float *&dist,
                 const int64_t *&ids, int64_t &result_size, float valid_ratio,
                 int index_type, FilterInterface *bitmap, bool reverse_filter,
                 bool need_extra_info, const char *&extra_infos,
                 void *&iter_ctx, bool is_last_search, void *allocator);
  int immutable_optimize();

  std::shared_ptr<vsag::Index> &get_index() { return index_; }
  void set_index(std::shared_ptr<vsag::Index> hnsw) { index_ = hnsw; }
  vsag::Allocator *get_allocator() const { return allocator_; }
  inline bool get_use_static() const { return use_static_; }
  inline int get_max_degree() const { return max_degree_; }
  inline int get_ef_construction() const { return ef_construction_; }
  inline int get_index_type() const { return (int)index_type_; }
  const char *get_dtype() const { return dtype_; }
  const char *get_metric() const { return metric_; }
  inline int get_ef_search() const { return ef_search_; }
  inline int get_dim() const { return dim_; }
  inline uint64_t get_extra_info_size() const { return extra_info_size_; }
  inline int16_t get_refine_type() const { return refine_type_; }
  inline int16_t get_bq_bits_query() const { return bq_bits_query_; }
  inline bool get_bq_use_fht() const { return bq_use_fht_; };
  inline bool get_use_reorder() const { return use_reorder_; }
  inline float get_doc_prune_ratio() const { return doc_prune_ratio_; }
  inline int get_window_size() const { return window_size_; }

  TO_STRING_KV(KP(this), K_(is_created), K_(is_build), K_(use_static), KCSTRING_(dtype),
      KCSTRING_(metric), K_(max_degree), K_(ef_construction), K_(ef_search), K_(dim),
      K_(ef_search), K_(index_type), KP(index_.get()), KP_(allocator), K_(extra_info_size),
      K_(refine_type), K_(bq_bits_query), K_(bq_use_fht));

private:
  uint64_t magic_;
  bool is_created_;
  bool is_build_;
  bool use_static_;
  const char *dtype_;
  const char *metric_;
  int max_degree_;
  int ef_construction_;
  int ef_search_;
  int dim_;
  IndexType index_type_;
  std::shared_ptr<vsag::Index> index_;
  vsag::Allocator *allocator_;
  uint64_t extra_info_size_;
  int16_t refine_type_;
  int16_t bq_bits_query_;
  bool bq_use_fht_;
  bool use_reorder_;
  float doc_prune_ratio_;
  int window_size_;
};

int HnswIndexHandler::build_index(const vsag::DatasetPtr &base)
{
  int ret = OB_SUCCESS;
  try {
    tl::expected<std::vector<int64_t>, Error> result = index_->Build(base);
    if (result.has_value()) {
    } else {
      ret = vsag_errcode2ob(result.error().type);
    }
  } catch (const std::exception &e) {
    ret = OB_ERR_VSAG_RETURN_ERROR;
    LOG_WARN("[OBVSAG] exception caught in build_index", "what", e.what());
  } catch (...) {
    ret = OB_ERR_VSAG_RETURN_ERROR;
    LOG_WARN("[OBVSAG] unknown exception caught in build_index");
  }
  return ret;
}

int HnswIndexHandler::get_index_number()
{
  return index_->GetNumElements();
}

int HnswIndexHandler::add_index(const vsag::DatasetPtr &incremental)
{
  int ret = OB_SUCCESS;
  tl::expected<std::vector<int64_t>, Error> result = index_->Add(incremental);
  if (result.has_value()) {
    LOG_DEBUG("add index success", K(get_index_number()));
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::cal_distance_by_id(uint32_t len, uint32_t *dims, float *vals,
                                         const int64_t *ids, int64_t count,
                                         const float *&dist)
{
  int ret = OB_SUCCESS;
  vsag::SparseVector sparse;
  sparse.len_ = len;
  sparse.ids_ = dims;
  sparse.vals_ = vals;
  DatasetPtr query = vsag::Dataset::Make();
  query->NumElements(1)->SparseVectors(&sparse)->Owner(false);
  float *dist_tmp = (float*)allocator_->Allocate(count * sizeof(float));
  if (OB_ISNULL(dist_tmp)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("failed to alloc memory for cal_distance", K(ret), K(count));
  }
  // TODO(ningxin.ning): support CalcDistanceById in sparse vector
  for (int i = 0; i < count && OB_SUCC(ret); ++i) {
    // tl::expected<float, Error> result = index_->CalcDistanceById(query, ids[i]);
    // if (result.has_value()) {
    //   dist_tmp[i] = result.value();
    // } else {
    //   ret = vsag_errcode2ob(result.error().type);
    // }
    dist_tmp[i] = 0.1;
  }
  dist = dist_tmp;
  return ret;
}

int HnswIndexHandler::cal_distance_by_id(const float *vector,
                                         const int64_t *ids, int64_t count,
                                         const float *&dist)
{
  int ret = OB_SUCCESS;
  tl::expected<DatasetPtr, Error> result = index_->CalDistanceById(vector, ids, count);
  if (result.has_value()) {
    result.value()->Owner(false);
    dist = result.value()->GetDistances();
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::get_extra_info_by_ids(const int64_t *ids, int64_t count,
                                            char *extra_infos)
{
  int ret = OB_SUCCESS;
  tl::expected<void, Error> result = index_->GetExtraInfoByIds(ids, count, extra_infos);
  if (result.has_value()) {
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::get_vid_bound(int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  int64_t element_cnt = index_->GetNumElements();
  if (element_cnt == 0) {
  } else {
    tl::expected<std::pair<int64_t, int64_t>, Error> result = index_->GetMinAndMaxId();
    if (result.has_value()) {
      min_vid = result.value().first;
      max_vid = result.value().second;
    } else {
      ret = vsag_errcode2ob(result.error().type);
    }
  }
  return ret;
}

uint64_t HnswIndexHandler::estimate_memory(const uint64_t row_count, const bool is_build)
{
  
  uint64_t size = 0;
  if (IPIVF_TYPE == index_type_) {
    // TODO(ningxin.ning): use vsag EstimateMemory
    size += 2 * sizeof(int64_t) * row_count;
    // nonzero dim = 100
    size += 100 * row_count * sizeof(float) * 2;
    if (use_reorder_) {
      size *= 2;
    }
  } else {
    size = index_->EstimateMemory(row_count);
  }
  if (HNSW_BQ_TYPE == index_type_ && is_build) {
    if (QuantizationType::SQ8 == refine_type_) {
      size += (row_count * dim_ * sizeof(uint8_t));
    } else {
      size += (row_count * dim_ * sizeof(float));
    }
  }
  return size;
}

int HnswIndexHandler::immutable_optimize()
{
  int ret = OB_SUCCESS;
  if (index_type_ == IPIVF_TYPE) {
    // TODO(ningxin.ning): support SetImmutable for sparse vector index
  } else {
    tl::expected<void, Error> res = index_->SetImmutable();
    if (res.has_value()) {
      LOG_INFO("[OBVSAG] set immutable success", KPC(this));
    } else {
      ret = vsag_errcode2ob(res.error().type);
      LOG_WARN("[OBVSAG] index set immutable error", K(ret), K(res.error().type));
    }
  }
  return ret;
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr &query, int64_t topk,
                                 const std::string &parameters,
                                 const float *&dist, const int64_t *&ids,
                                 int64_t &result_size, float valid_ratio,
                                 int index_type, FilterInterface *bitmap,
                                 bool reverse_filter, bool need_extra_info,
                                 const char *&extra_infos, void *allocator,
                                 float distance_threshold)
{
  int ret = OB_SUCCESS;
  std::function<bool(int64_t)> vid_filter = [bitmap, reverse_filter](int64_t id) -> bool {
    if (!reverse_filter) {
      return bitmap->test(id);
    } else {
      return !(bitmap->test(id));
    }
  };
  std::function<bool(const char *)> exinfo_filter = [bitmap, reverse_filter](const char *data) -> bool {
    if (!reverse_filter) {
      return bitmap->test(data);
    } else {
      return !(bitmap->test(data));
    }
  };

  std::shared_ptr<ObVasgFilter> vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, vid_filter, exinfo_filter);
  vsag::Allocator *vsag_allocator = nullptr;
  if (allocator != nullptr) vsag_allocator = static_cast<vsag::Allocator *>(allocator);
  tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result;
  if (index_type_ == IPIVF_TYPE) {
    result = index_->KnnSearch(query, topk, parameters, bitmap == nullptr ? nullptr : vsag_filter);
  } else {
    vsag::SearchParam search_param(false, parameters,
                                 bitmap == nullptr ? nullptr : vsag_filter,
                                 vsag_allocator);
    result = index_->KnnSearch(query, topk, search_param);
  }
  if (result.has_value()) {
    // the lifecycle of result
    result.value()->Owner(false);
    ids = result.value()->GetIds();
    dist = result.value()->GetDistances();
    result_size = result.value()->GetDim();
    if (need_extra_info) {
      extra_infos = result.value()->GetExtraInfos();
    }
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr &query, int64_t topk,
                                 const std::string &parameters,
                                 const float *&dist, const int64_t *&ids,
                                 int64_t &result_size, float valid_ratio,
                                 int index_type, FilterInterface *bitmap,
                                 bool reverse_filter, bool need_extra_info,
                                 const char *&extra_infos, void *&iter_ctx,
                                 bool is_last_search, void *allocator)
{
  int ret = OB_SUCCESS;
  std::function<bool(int64_t)> filter = [bitmap, reverse_filter](int64_t id) -> bool {
    if (!reverse_filter) {
      return bitmap->test(id);
    } else {
      return !(bitmap->test(id));
    }
  };
  std::function<bool(const char *)> exinfo_filter = [bitmap, reverse_filter](const char *data) -> bool {
    if (!reverse_filter) {
      return bitmap->test(data);
    } else {
      return !(bitmap->test(data));
    }
  };

  std::shared_ptr<ObVasgFilter> vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, filter, exinfo_filter);
  vsag::Allocator *vsag_allocator = nullptr;
  if (allocator != nullptr) vsag_allocator = static_cast<vsag::Allocator *>(allocator);
  vsag::IteratorContext *input_iter = static_cast<vsag::IteratorContext *>(iter_ctx);
  vsag::SearchParam search_param(true, parameters,
                                 bitmap == nullptr ? nullptr : vsag_filter,
                                 vsag_allocator, input_iter, is_last_search);
  tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result = index_->KnnSearch(query, topk, search_param);
  if (result.has_value()) {
    iter_ctx = search_param.iter_ctx;
    result.value()->Owner(false);
    ids = result.value()->GetIds();
    dist = result.value()->GetDistances();
    result_size = result.value()->GetDim();
    if (need_extra_info) {
      extra_infos = result.value()->GetExtraInfos();
    }
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

void set_log_level(int32_t ob_level_num)
{
  static std::map<int32_t, int32_t> ob2vsag_log_level = {
      {0 /*ERROR*/, vsag::Logger::Level::kERR},
      {1 /*WARN*/, vsag::Logger::Level::kWARN},
      {2 /*INFO*/, vsag::Logger::Level::kINFO},
      {3 /*EDIAG*/, vsag::Logger::Level::kERR},
      {4 /*WDIAG*/, vsag::Logger::Level::kWARN},
      {5 /*TRACE*/, vsag::Logger::Level::kTRACE},
      {6 /*DEBUG*/, vsag::Logger::Level::kDEBUG},
  };
  vsag::Options::Instance().logger()->SetLevel(
      static_cast<vsag::Logger::Level>(ob2vsag_log_level[ob_level_num]));
}

bool is_init_ = vsag::init();
bool is_init()
{
    LOG_INFO("[OBVSAG] Init VsagLib]:");
    if (is_init_) {
        LOG_INFO("[OBVSAG] Init VsagLib success");
    } else {
        LOG_INFO("[OBVSAG] Init VsagLib fail");
    }
    return is_init_; 
}

void set_logger(void *logger_ptr)
{
  vsag::Options::Instance().set_logger(static_cast<vsag::Logger *>(logger_ptr));
  vsag::Logger::Level log_level = static_cast<vsag::Logger::Level>(1); // default is debug level
  vsag::Options::Instance().logger()->SetLevel(log_level);
}

void set_block_size_limit(uint64_t size)
{
  vsag::Options::Instance().set_block_size_limit(size);
}

bool get_is_hgraph_type(uint8_t create_type) 
{
  bool res = false;
  switch (create_type) {
    case HNSW_TYPE: {
      res = false;
      break;
    }
    case HNSW_SQ_TYPE: 
    case HNSW_BQ_TYPE:
    case HGRAPH_TYPE: {
      res = true;
      break;
    }
  }
  return res;
}

const char* get_index_type_str(uint8_t create_type)
{
  const char* res;
  switch (create_type) {
    case HNSW_TYPE: {
      res = "hnsw";
      break;
    }
    case HNSW_SQ_TYPE: 
    case HNSW_BQ_TYPE:
    case HGRAPH_TYPE: {
      res = "hgraph";
      break;
    }
    case IPIVF_TYPE: {
      res = "sindi";
      break;
    }
  }
  return res;
}

const char* get_precise_quantization_type(const uint8_t type)
{
  const char* res = nullptr;
  if (type == QuantizationType::SQ8) {
    res = "sq8";
  } else {
    res = "fp32";
  }
  return res;
}


/**
  eg:
    hnsw: {
            "dtype": dtype, "metric_type": metric, "dim": dim, 
            "hnsw": {
              "max_degree": max_degree, "ef_construction": ef_construction, "ef_search": ef_search, "use_static": use_static
            }
          }
    hgraph: {
              "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
              "index_param": {
                "base_quantization_type": "fp32", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0
              }
            }
    sq: {
          "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
          "index_param": {
            "base_quantization_type": "sq8", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0
          }
        }
    bq: {
          "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
          "index_param": {
            "base_quantization_type": "rabitq", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0,
            "use_reorder": true, "ignore_reorder": true, "precise_quantization_type": "fp32", "precise_io_type": "block_memory_io"
          }
        }
*/
int construct_vsag_create_param(
    uint8_t create_type, const char *dtype, const char *metric, int dim,
    int max_degree, int ef_construction, int ef_search, void *allocator,
    int extra_info_size, int16_t refine_type, int16_t bq_bits_query,
    bool bq_use_fht, char *result_param_str)
{
  int ret = OB_SUCCESS;
  bool is_hgraph_type = get_is_hgraph_type(create_type);
  const char *index_type_str = is_hgraph_type ? "index_param" : "hnsw";
  const char *base_quantization_type;
  const int64_t buf_len = 1024;
  switch (create_type) {
  case HNSW_SQ_TYPE: {
    base_quantization_type = "sq8";
    break;
  }
  case HNSW_BQ_TYPE: {
    base_quantization_type = "rabitq";
    break;
  }
  case HGRAPH_TYPE: {
    base_quantization_type = "fp32";
    break;
  }
  default: {
    break;
  }
  }
  // ObIStreamBuf only supports seeking within the current callback buffer, while
  // VSAG's new format seeks to the footer. Keep the legacy format until global seek is supported.
  int64_t pos = 0;
  int64_t buff_size = 0;
  if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "{\"dim\":%d",
                              int(dim)))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"dtype\":\"%s\"",
                                     dtype))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"metric_type\":\"%s\"",
                                     metric))) {
  } else if (extra_info_size > 0 &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"extra_info_size\": %d",
                                 extra_info_size))) {
    LOG_WARN("failed to fill result_param_str", K(ret), K(extra_info_size));
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"use_old_serial_format\":true"))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"%s\":{",
                                     index_type_str))) {
  } else if (OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos, "\"ef_construction\":%d",
                 ef_construction))) {
  } else if (! is_hgraph_type && OB_FAIL(databuff_printf(result_param_str,
                                 buf_len, pos, ",\"ef_search\":%d",
                                 ef_search))) {
    LOG_WARN("failed to fill result_param_str", K(ret), K(ef_search));
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"max_degree\":%d",
                                     max_degree))) {
  } else if (is_hgraph_type &&
      OB_FAIL(databuff_printf(
          result_param_str, buf_len, pos,
          ",\"base_quantization_type\":\"%s\"",
          base_quantization_type))) {
    LOG_WARN("failed to fill result_param_str", K(ret), K(base_quantization_type));
  } else if (is_hgraph_type &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"build_thread_count\":%d",
                                     0))) {
    LOG_WARN("failed to fill result_param_str", K(ret));
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"use_reorder\":true"))) {
    LOG_WARN("failed to fill result_param_str", K(ret));
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"ignore_reorder\":true"))) {
    LOG_WARN("failed to fill result_param_str", K(ret));
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"precise_quantization_type\":\"%s\"", get_precise_quantization_type(refine_type)))) {
    LOG_WARN("failed to fill result_param_str", K(ret), K(refine_type));
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"precise_io_type\":\"block_memory_io\""))) {
    LOG_WARN("failed to fill result_param_str", K(ret));
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"rabitq_bits_per_dim_query\":%d", bq_bits_query))) {
    LOG_WARN("failed to fill result_param_str", K(ret), K(bq_bits_query));
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"rabitq_use_fht\":%s", (bq_use_fht ? "true" : "false")))) {
    LOG_WARN("failed to fill result_param_str", K(ret), K(bq_use_fht));
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_INFO("build param", K(create_type), KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

int construct_vsag_sindi_create_param(uint8_t create_type, const char *dtype, const char *metric, 
    void *allocator, int extra_info_size, bool use_reorder, float doc_prune_ratio, int window_size,
    char *result_param_str)
{
  int ret = OB_SUCCESS;
  const char *index_type_str = "index_param";
  const int64_t buf_len = 1024;

  int64_t pos = 0;
  int64_t buff_size = 0;
  // ObIStreamBuf exposes the serialized index through callback-backed chunks.
  // Skip seek-based footer handling and let SINDI read from that stream directly;
  // BufferStreamReader otherwise treats the current chunk length as the full stream.
  const bool deserialize_without_footer = true;
  const bool deserialize_without_buffer = true;
  if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "{\"dtype\":\"%s\"", dtype))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"metric_type\":\"%s\"", metric))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"dim\": 1024"))) {
  } else if (extra_info_size > 0 &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"extra_info_size\": %d", extra_info_size))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos, "\"use_reorder\":%s", use_reorder ? "true" : "false"))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"doc_prune_ratio\":%f", doc_prune_ratio))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"window_size\":%d", window_size))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"deserialize_without_footer\":%s",
                                 (deserialize_without_footer ? "true": "false")))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"deserialize_without_buffer\":%s",
                                 (deserialize_without_buffer ? "true": "false")))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_INFO("build param", K(create_type), KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

/**
  eg:
    hnsw : {"hnsw": {"ef_search": ef_search, "skip_ratio": 0.7}}
    hgraph : {"hgraph": {"ef_search": ef_search, "use_extra_info_filter": use_extra_info_filter}}
*/
int construct_vsag_search_param(uint8_t create_type, 
                                int64_t ef_search, 
                                bool use_extra_info_filter, 
                                char *result_param_str)
{
  int ret = OB_SUCCESS;
  bool is_hgraph_type = get_is_hgraph_type(create_type);
  const char *index_type_str = is_hgraph_type ? "hgraph" : "hnsw";
  int64_t pos = 0;
  int64_t buff_size = 0;
  int64_t buf_len = 1024;
  if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "{\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "\"ef_search\":%d", int(ef_search)))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"skip_ratio\":%f", 0.7))) {
  } else if (is_hgraph_type && OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"use_extra_info_filter\":%s", use_extra_info_filter ? "true" : "false"))) {
    LOG_WARN("failed to fill result_param_str", K(ret), K(index_type_str));
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_TRACE("search param", KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

int construct_vsag_sindi_search_param(float query_prune_ratio, uint64_t n_candidate, 
                                char *result_param_str)
{
  int ret = OB_SUCCESS;
  const char *index_type_str = "sindi";
  int64_t pos = 0;
  int64_t buff_size = 0;
  int64_t buf_len = 1024;
  if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "{\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "\"query_prune_ratio\":%f", query_prune_ratio))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"n_candidate\":%lu}}", n_candidate))) {
  }
  if (OB_SUCC(ret)) {
    LOG_TRACE("search param", KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

int create_index(VectorIndexPtr &index_handler,
                 IndexType index_type, const char *dtype,
                 const char *metric, int dim, int max_degree,
                 int ef_construction, int ef_search, void *allocator,
                 int extra_info_size /* = 0*/, int16_t refine_type /*= 0*/,
                 int16_t bq_bits_query /*= 32*/, bool bq_use_fht /*= false*/)
{
  int ret = OB_SUCCESS;
  if (dtype == nullptr || metric == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer", KP(dtype), KP(metric));
  } else {
    vsag::Allocator *vsag_allocator = nullptr;
    if (allocator == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(index_type), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(allocator);
      LOG_INFO("[OBVSAG] use caller allocator ", K(index_type), K(lbt()));
    }
  
    adjust_create_index_max_degree(index_type, max_degree);

    const char* index_type_str = get_index_type_str(index_type);
    char result_param_str[1024] = {0};
    if (OB_FAIL(construct_vsag_create_param(
        uint8_t(index_type), dtype, metric, dim, max_degree, 
        ef_construction, ef_search, allocator, extra_info_size,
        refine_type, bq_bits_query, bq_use_fht, result_param_str))) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index = vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (index.has_value()) {
        std::shared_ptr<vsag::Index> hnsw;
        hnsw = index.value();
        HnswIndexHandler *hnsw_index = new HnswIndexHandler(
            true, false, false, dtype, metric, max_degree, ef_construction,
            ef_search, dim, index_type, hnsw, vsag_allocator, extra_info_size,
            refine_type, bq_bits_query, bq_use_fht);
        if (OB_ISNULL(hnsw_index)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
          LOG_WARN("new HnswIndexHandler fail", K(ret), K(index_type));
        } else {
          index_handler = static_cast<VectorIndexPtr>(hnsw_index);
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBVSAG] create index error happend",
            K(ret), KCSTRING(result_param_str), K(index.error().type), KCSTRING(index.error().message.c_str()));
      }
    }
  }
  return ret;
}

int validate_create_index(const CreateIndexParam &param, std::string &err_msg)
{
  int ret = OB_SUCCESS;
  err_msg.clear();
  if (param.dtype_ == nullptr || param.metric_ == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer", KP(param.dtype_), KP(param.metric_));
  } else {
    vsag::Allocator *vsag_allocator = nullptr;
    if (param.allocator_ == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(param.index_type_), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(param.allocator_);
      LOG_INFO("[OBVSAG] use caller allocator ", K(param.index_type_), K(lbt()));
    }

    const char *index_type_str = get_index_type_str(param.index_type_);
    char result_param_str[1024] = {0};
    if (param.is_sparse_) {
      if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(param.index_type_),
                                                    param.dtype_,
                                                    param.metric_,
                                                    param.allocator_,
                                                    param.extra_info_size_,
                                                    param.use_reorder_,
                                                    param.doc_prune_ratio_,
                                                    param.window_size_,
                                                    result_param_str))) {
      }
    } else {
      int max_degree = param.max_degree_;
      adjust_create_index_max_degree(param.index_type_, max_degree);
      if (OB_FAIL(construct_vsag_create_param(
          uint8_t(param.index_type_), param.dtype_, param.metric_, param.dim_, max_degree,
          param.ef_construction_, param.ef_search_, param.allocator_, param.extra_info_size_,
          param.refine_type_, param.bq_bits_query_, param.bq_use_fht_, result_param_str))) {
      }
    }
    if (OB_SUCC(ret)) {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index =
          vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (!index.has_value()) {
        ret = vsag_errcode2ob(index.error().type);
        fill_vsag_error_message(index.error(), err_msg);
        LOG_WARN("[OBVSAG] validate create index error",
            K(ret), KCSTRING(result_param_str), K(index.error().type), KCSTRING(index.error().message.c_str()));
      }
    }
  }
  return ret;
}

int create_index(VectorIndexPtr &index_handler, IndexType index_type, const char *dtype, const char *metric,
    bool use_reorder, float doc_prune_ratio, int window_size, void *allocator, int extra_info_size /* = 0*/)
{
  int ret = OB_SUCCESS;
  if (dtype == nullptr || metric == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer", KP(dtype), KP(metric));
  } else {
    vsag::Allocator *vsag_allocator = nullptr;
    if (allocator == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(index_type), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(allocator);
      LOG_INFO("[OBVSAG] use caller allocator ", K(index_type), K(lbt()));
    }

    const char *index_type_str = get_index_type_str(index_type);
    char result_param_str[1024] = {0};
    if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(index_type),
            dtype,
            metric,
            allocator,
            extra_info_size,
            use_reorder,
            doc_prune_ratio,
            window_size,
            result_param_str))) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index =
          vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (index.has_value()) {
        std::shared_ptr<vsag::Index> hnsw;
        hnsw = index.value();
        HnswIndexHandler *hnsw_index = new HnswIndexHandler(true,
            false,
            false,
            dtype,
            metric,
            index_type,
            hnsw,
            vsag_allocator,
            extra_info_size,
            use_reorder,
            doc_prune_ratio,
            window_size);
        if (OB_ISNULL(hnsw_index)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
          LOG_WARN("new HnswIndexHandler fail", K(ret), K(index_type));
        } else {
          index_handler = static_cast<VectorIndexPtr>(hnsw_index);
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBVSAG] create index error happend", K(ret), KCSTRING(result_param_str), K(index.error().type));
      }
    }
  }
  return ret;
}

int build_index(VectorIndexPtr &index_handler, float *vector_list,
                int64_t *ids, int dim, int size, char *extra_infos /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || vector_list == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", KP(index_handler), KP(vector_list), K(ids));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    DatasetPtr dataset = vsag::Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(size)
        ->Ids(ids)
        ->Float32Vectors(vector_list)
        ->Owner(false);
    if (extra_infos != nullptr) {
      dataset->ExtraInfos(extra_infos);
    }
    if (OB_FAIL(hnsw->build_index(dataset))) {
    }
  }
  return ret;
}

int build_index(VectorIndexPtr &index_handler, uint32_t *lens, uint32_t *dims, float *vals, int64_t *ids, int size,
    char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || lens == nullptr || dims == nullptr || vals == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", KP(index_handler), KP(lens), KP(dims), KP(vals), KP(vals), KP(ids));
  } else {
    uint32_t *cur_dims_ptr = dims;
    float *cur_vals_ptr = vals;
    std::vector<vsag::SparseVector> sparse_vectors(size);
    for (int i = 0; i < size; i++) {
      sparse_vectors[i].len_ = lens[i];
      sparse_vectors[i].ids_ = cur_dims_ptr;
      sparse_vectors[i].vals_ = cur_vals_ptr;
      cur_dims_ptr += lens[i];
      cur_vals_ptr += lens[i];
    }
    HnswIndexHandler *handler = static_cast<HnswIndexHandler *>(index_handler);
    DatasetPtr dataset = vsag::Dataset::Make();
    dataset->NumElements(size)->Ids(ids)->SparseVectors(sparse_vectors.data())->Owner(false);
    if (extra_info != nullptr) {
      dataset->ExtraInfos(extra_info);
    }
    if (OB_FAIL(handler->build_index(dataset))) {
    }
  }
  return ret;
}

int add_index(VectorIndexPtr &index_handler, float *vector,
              int64_t *ids, int dim, int size,
              char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || vector == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", KP(index_handler), KP(vector), KP(ids));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    // add index
    DatasetPtr incremental = vsag::Dataset::Make();
    incremental->Dim(dim)
        ->NumElements(size)
        ->Ids(ids)
        ->Float32Vectors(vector)
        ->Owner(false);
    if (extra_info != nullptr) {
      incremental->ExtraInfos(extra_info);
    }
    if (OB_FAIL(hnsw->add_index(incremental))) {
    }
  }
  return ret;
}

int add_index(VectorIndexPtr &index_handler, uint32_t *lens, uint32_t *dims, float *vals, int64_t *ids, int size,
    char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || lens == nullptr || dims == nullptr || vals == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", KP(index_handler), KP(lens), KP(dims), KP(vals), KP(vals), KP(ids));
  } else {
    uint32_t *cur_dims_ptr = dims;
    float *cur_vals_ptr = vals;
    std::vector<vsag::SparseVector> sparse_vectors(size);
    for (int i = 0; i < size; i++) {
      sparse_vectors[i].len_ = lens[i];
      sparse_vectors[i].ids_ = cur_dims_ptr;
      sparse_vectors[i].vals_ = cur_vals_ptr;
      cur_dims_ptr += lens[i];
      cur_vals_ptr += lens[i];
    }
    const uint32_t MAX_DIM_LIMIT = 500000;
    uint32_t max_dim = 0;
    for (int i = 0; i < size && OB_SUCC(ret); i++) {
      uint32_t length = sparse_vectors[i].len_;
      for (int j = 0; j < length && OB_SUCC(ret); j++) {
        max_dim = MAX(max_dim, sparse_vectors[i].ids_[j]);
        if (OB_UNLIKELY(max_dim > MAX_DIM_LIMIT)) {
          ret = OB_NOT_SUPPORTED;
          LOG_WARN("sparse vector dimension greater than 500000 is not supported.", K(ret), K(max_dim));
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "sparse vector dimension greater than 500000 is");
        }
      }
    }
    if (OB_FAIL(ret)) {
      // do nothing
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
      // add index
      DatasetPtr incremental = vsag::Dataset::Make();
      incremental->NumElements(size)->Ids(ids)->SparseVectors(sparse_vectors.data())->Owner(false);
      if (extra_info != nullptr) {
        incremental->ExtraInfos(extra_info);
      }
      if (OB_FAIL(hnsw->add_index(incremental))) {
      }
    }
  }
  return ret;
}

int get_index_type(VectorIndexPtr &index_handler)
{
  HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
  return hnsw->get_index_type();
}

int get_index_number(VectorIndexPtr &index_handler, int64_t &size)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(index_handler));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    size = hnsw->get_index_number();
  }
  return ret;
}

int cal_distance_by_id(VectorIndexPtr &index_handler,
                       const float *vector, const int64_t *ids, int64_t count,
                       const float *&distances)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(index_handler));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->cal_distance_by_id(vector, ids, count, distances))) {
    }
  }
  return ret;
}

int cal_distance_by_id(VectorIndexPtr &index_handler, uint32_t len, uint32_t *dims, float *vals, const int64_t *ids,
    int64_t count, const float *&distances)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(index_handler));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->cal_distance_by_id(len, dims, vals, ids, count, distances))) {
    }
  }
  return ret;
}

int get_vid_bound(VectorIndexPtr &index_handler,
                         int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  if (nullptr == index_handler) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", KP(index_handler));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
    if (index_type == IPIVF_TYPE) {
      // TODO(ningxin.ning): support get_vid_bound for ipivf
      min_vid = 0;
      max_vid = 0;
    } else {
      if (OB_FAIL(hnsw->get_vid_bound(min_vid, max_vid))) {
      }
    }
  }
  return ret;
}

int knn_search(VectorIndexPtr &index_handler, float *query_vector,
               int dim, int64_t topk, const float *&dist, const int64_t *&ids,
               int64_t &result_size, int ef_search, bool need_extra_info,
               const char *&extra_infos, void *invalid, bool reverse_filter,
               bool use_extra_info_filter, void *allocator, float valid_ratio, 
               float distance_threshold)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || query_vector == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", KP(index_handler), KP(query_vector));
  } else {
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
    char result_param_str[1024]= {0};
    const int64_t EF_SEARCH_LIMIT = 1000L;
    const int64_t AMPLIFICATION_FACTOR = 10;
    if (ef_search > EF_SEARCH_LIMIT) {
      int64_t index_number = hnsw->get_index_number();
      if (0 != index_number) {
        topk = topk < index_number ? topk : index_number;
      }
      int64_t ef_search_threshold = AMPLIFICATION_FACTOR * topk > EF_SEARCH_LIMIT ? AMPLIFICATION_FACTOR * topk : EF_SEARCH_LIMIT;
      ef_search = ef_search < ef_search_threshold ? ef_search : ef_search_threshold;
    }
    if (OB_FAIL(construct_vsag_search_param(uint8_t(index_type), ef_search, use_extra_info_filter, result_param_str))) {
    } else {
      const std::string input_json_string(result_param_str);
      DatasetPtr query = vsag::Dataset::Make();
      query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
      if (OB_FAIL(hnsw->knn_search(query, topk, input_json_string, dist, ids,
                                   result_size, valid_ratio, index_type, bitmap,
                                   reverse_filter, need_extra_info, extra_infos, allocator, distance_threshold))) {
      }
    }
  }
  return ret;
}

int knn_search(VectorIndexPtr &index_handler, float *query_vector,
               int dim, int64_t topk, const float *&dist, const int64_t *&ids,
               int64_t &result_size, int ef_search, bool need_extra_info,
               const char *&extra_infos, void *invalid, bool reverse_filter,
               bool use_extra_info_filter, float valid_ratio, void *&iter_ctx,
               bool is_last_search, void *allocator)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || query_vector == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(index_handler), K(query_vector));
  } else {
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
    char result_param_str[1024]= {0};
    const int64_t EF_SEARCH_LIMIT = 1000L;
    const int64_t AMPLIFICATION_FACTOR = 10;
    if (ef_search > EF_SEARCH_LIMIT) {
      int64_t index_number = hnsw->get_index_number();
      if (0 != index_number) {
        topk = topk < index_number ? topk : index_number;
      }
      int64_t ef_search_threshold = AMPLIFICATION_FACTOR * topk > EF_SEARCH_LIMIT ? AMPLIFICATION_FACTOR * topk : EF_SEARCH_LIMIT;
      ef_search = ef_search < ef_search_threshold ? ef_search : ef_search_threshold;
    }
    if (OB_FAIL(construct_vsag_search_param(uint8_t(index_type), ef_search, use_extra_info_filter, result_param_str))) {
    } else {
      const std::string input_json_string(result_param_str);
      DatasetPtr query = vsag::Dataset::Make();
      query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
      if (OB_FAIL(hnsw->knn_search(query, topk, input_json_string, dist, ids,
                            result_size, valid_ratio, index_type, bitmap,
                            reverse_filter, need_extra_info, extra_infos, iter_ctx,
                            is_last_search, allocator))) {
      }
    }
  }
  return ret;
}

int knn_search(obvsag::VectorIndexPtr &index_handler, uint32_t len, uint32_t *dims, float *vals, int64_t topk,
    const float *&result_dist, const int64_t *&result_ids, const char *&extra_infos, int64_t &result_size,
    float query_prune_ratio, int64_t n_candidate, void *invalid, bool reverse_filter,
    bool is_extra_info_filter, float valid_ratio, void *allocator, bool need_extra_info)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || dims == nullptr || vals == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(index_handler), K(dims), K(vals));
  } else {
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
    char result_param_str[1024]= {0};
    if (OB_FAIL(construct_vsag_sindi_search_param(query_prune_ratio, n_candidate, result_param_str))) {
    } else if (len == 0) {
      result_size = 0;
    } else {
      const std::string input_json_string(result_param_str);
      vsag::SparseVector sparse;
      sparse.len_ = len;
      sparse.ids_ = dims;
      sparse.vals_ = vals;
      DatasetPtr query = vsag::Dataset::Make();
      query->NumElements(1)->SparseVectors(&sparse)->Owner(false);
      if (OB_FAIL(hnsw->knn_search(query, topk, input_json_string, result_dist, result_ids,
                            result_size, valid_ratio, index_type, bitmap,
                            reverse_filter, need_extra_info, extra_infos, allocator))) {
      }
    }
  }
  return ret;
}

int fserialize(VectorIndexPtr &index_handler, std::ostream &out_stream)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(index_handler));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    tl::expected<void, Error> bs = hnsw->get_index()->Serialize(out_stream);
    if (bs.has_value()) {
      LOG_INFO("[OBVSAG] serialize index success");
    } else {
      ret = vsag_errcode2ob(bs.error().type);
      LOG_WARN("[OBVSAG] fserialize error happend", K(ret), K(bs.error().type));
    }
  }
  return ret;
}

int fdeserialize(VectorIndexPtr &index_handler,
                 std::istream &in_stream)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(index_handler));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    std::shared_ptr<vsag::Index> hnsw_index;
    bool use_static = hnsw->get_use_static();
    const char *metric = hnsw->get_metric();
    const char *dtype = hnsw->get_dtype();
    int max_degree = hnsw->get_max_degree();
    int ef_construction = hnsw->get_ef_construction();
    int ef_search = hnsw->get_ef_search();
    int dim = hnsw->get_dim();
    int index_type = hnsw->get_index_type();
    uint64_t extra_info_size = hnsw->get_extra_info_size();
    const char* index_type_str = get_index_type_str(index_type);
    int16_t refine_type = hnsw->get_refine_type();
    int16_t bq_bits_query = hnsw->get_bq_bits_query();
    bool bq_use_fht = hnsw->get_bq_use_fht();
    bool use_reorder = hnsw->get_use_reorder();
    float doc_prune_ratio = hnsw->get_doc_prune_ratio();
    int window_size = hnsw->get_window_size();

    char result_param_str[1024] = {0};
    if ((IndexType)index_type == IndexType::IPIVF_TYPE) {
      if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(index_type),
              dtype,
              metric,
              hnsw->get_allocator(),
              extra_info_size,
              use_reorder,
              doc_prune_ratio,
              window_size,
              result_param_str))) {
      }
    } else {
      if (OB_FAIL(construct_vsag_create_param(
        uint8_t(index_type), dtype, metric, dim, max_degree,
        ef_construction, ef_search, hnsw->get_allocator(),
        extra_info_size, refine_type, bq_bits_query, bq_use_fht, result_param_str))) {
      } 
    }
    if (OB_FAIL(ret)) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index = vsag::Factory::CreateIndex(index_type_str, input_json_str, hnsw->get_allocator());
      if (index.has_value()) {
        hnsw_index = index.value();
        tl::expected<void, Error> bs = hnsw_index->Deserialize(in_stream);
        if (bs.has_value()) {
          hnsw->set_index(hnsw_index);
          LOG_INFO("[OBVSAG] fdeserialize success", KCSTRING(result_param_str));
        } else {
          ret = vsag_errcode2ob(bs.error().type);
          LOG_WARN("[OBVSAG] fdeserialize error", K(ret), K(bs.error().type));
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBVSAG] create index error", K(ret), K(index.error().type));
      }
    }
  }
  return ret;
}

int delete_index(VectorIndexPtr &index_handler)
{
  int ret = OB_SUCCESS;
  LOG_INFO("[OBVSAG] delete index ",
      KP((void *)static_cast<HnswIndexHandler *>(index_handler)->get_index().get()),
      K(static_cast<HnswIndexHandler *>(index_handler)->get_index().use_count()), K(lbt()));
  if (index_handler != nullptr) {
    delete static_cast<HnswIndexHandler *>(index_handler);
    index_handler = nullptr;
  }
  return ret;
}

void delete_iter_ctx(void *iter_ctx)
{
  LOG_TRACE("[OBVAG] delete_iter_ctx", KP(iter_ctx), K(lbt()));
  if (iter_ctx != nullptr) {
    delete static_cast<vsag::IteratorContext *>(iter_ctx);
    iter_ctx = nullptr;
  }
}

int get_extra_info_by_ids(VectorIndexPtr &index_handler,
                          const int64_t *ids, int64_t count,
                          char *extra_infos)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(index_handler));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->get_extra_info_by_ids(ids, count, extra_infos))) {
    }
  }
  return ret;
}

uint64_t estimate_memory(VectorIndexPtr &index_handler, const uint64_t row_count, const bool is_build) 
{
  uint64_t estimate_memory_size = 0;
  if (index_handler != nullptr) {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    estimate_memory_size = hnsw->estimate_memory(row_count, is_build);
  }
  return estimate_memory_size;
}

int immutable_optimize(VectorIndexPtr& index_handler)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index_handler)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[OBVSAG] null pointer addr", K(ret), KP(index_handler));
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->immutable_optimize())) {
    }
  }
  return ret;
}

} // namespace obvsag
} // namespace common
} // namespace oceanbase

#ifdef OB_ENABLE_CUVS
namespace oceanbase {
namespace common {
namespace obcuvs {

namespace {

static int cuvs_error(const char *where)
{
  const char *msg = cuvsGetLastErrorText();
  int ret = OB_ERR_VSAG_RETURN_ERROR;
  LOG_WARN("cuVS call failed", K(where), KP(msg), K(ret));
  return ret;
}

static int check_cuvs(cuvsError_t status, const char *where)
{
  return status == CUVS_SUCCESS ? OB_SUCCESS : cuvs_error(where);
}

static bool parse_metric(const char *metric, cuvsDistanceType &result)
{
  if (metric == nullptr) {
    return false;
  } else if (0 == strcasecmp(metric, "l2")) {
    result = L2Expanded;
  } else if (0 == strcasecmp(metric, "cosine")) {
    result = CosineExpanded;
  } else if (0 == strcasecmp(metric, "ip") || 0 == strcasecmp(metric, "inner_product")) {
    result = InnerProduct;
  } else {
    return false;
  }
  return true;
}

static void init_tensor(DLManagedTensor &tensor, void *data, int64_t *shape,
                        int ndim, DLDataType dtype)
{
  memset(&tensor, 0, sizeof(tensor));
  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.device.device_id = 0;
  tensor.dl_tensor.ndim = ndim;
  tensor.dl_tensor.dtype = dtype;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;
}

static DLDataType float32_dtype()
{
  DLDataType dtype;
  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;
  return dtype;
}

static DLDataType uint64_dtype()
{
  DLDataType dtype;
  dtype.code = kDLUInt;
  dtype.bits = 64;
  dtype.lanes = 1;
  return dtype;
}

class TempFile {
public:
  TempFile() : fd_(-1)
  {
    char name[] = "/tmp/seekdb-cuvs-XXXXXX";
    fd_ = mkstemp(name);
    if (fd_ >= 0) {
      path_ = name;
      close(fd_);
      fd_ = -1;
    }
  }
  ~TempFile() { if (!path_.empty()) { unlink(path_.c_str()); } }
  bool valid() const { return !path_.empty(); }
  const char *path() const { return path_.c_str(); }
private:
  int fd_;
  std::string path_;
};

class CuvsIndexHandler {
public:
  CuvsIndexHandler(int &ret, obvsag::IndexType index_type, const char *dtype,
                   const char *metric, int dim, int max_degree,
                   int ef_construction, int ef_search, int extra_info_size)
      : magic_(CUVS_HANDLER_MAGIC), built_(false), immutable_(false), dim_(dim),
        max_degree_(max_degree), ef_construction_(ef_construction),
        ef_search_(ef_search), index_type_(index_type),
        extra_info_size_(extra_info_size), resources_(0), index_(nullptr),
        index_params_(nullptr), ace_params_(nullptr), search_params_(nullptr)
  {
    ret = OB_SUCCESS;
    if (!parse_metric(metric, metric_)) {
      ret = OB_NOT_SUPPORTED;
    } else if (0 != strcasecmp(dtype == nullptr ? "" : dtype, "float32")) {
      ret = OB_NOT_SUPPORTED;
    } else if (OB_FAIL(check_cuvs(cuvsResourcesCreate(&resources_), "resources"))) {
    } else if (OB_FAIL(check_cuvs(cuvsHnswAceParamsCreate(&ace_params_), "ace params"))) {
    } else if (OB_FAIL(check_cuvs(cuvsHnswIndexParamsCreate(&index_params_), "index params"))) {
    } else if (OB_FAIL(check_cuvs(cuvsHnswIndexCreate(&index_), "index"))) {
    } else if (OB_FAIL(check_cuvs(cuvsHnswSearchParamsCreate(&search_params_), "search params"))) {
    } else {
      index_params_->hierarchy = CPU;
      index_params_->M = static_cast<size_t>(max_degree_);
      index_params_->ef_construction = ef_construction_;
      index_params_->metric = metric_;
      index_params_->ace_params = ace_params_;
      search_params_->ef = ef_search_;
      search_params_->num_threads = 0;
    }
    if (OB_FAIL(ret)) {
      cleanup();
    }
  }

  ~CuvsIndexHandler() { cleanup(); }

  int build(float *vectors, int64_t *ids, int dim, int size, char *extra_infos)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (vectors == nullptr || ids == nullptr || dim != dim_ || size < 0) {
      return OB_INVALID_ARGUMENT;
    }
    if (built_) {
      return OB_ERR_UNEXPECTED;
    }
    int64_t shape[2] = {size, dim};
    DLManagedTensor dataset;
    init_tensor(dataset, vectors, shape, 2, float32_dtype());
    int ret = check_cuvs(cuvsHnswBuild(resources_, index_params_, &dataset, index_), "build");
    if (OB_SUCC(ret)) {
      copy_rows(vectors, ids, extra_infos, size);
      built_ = true;
    }
    return ret;
  }

  int add(float *vectors, int64_t *ids, int dim, int size, char *extra_infos)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (vectors == nullptr || ids == nullptr || dim != dim_ || size < 0) {
      return OB_INVALID_ARGUMENT;
    }
    if (!built_) {
      int64_t shape[2] = {size, dim};
      DLManagedTensor dataset;
      init_tensor(dataset, vectors, shape, 2, float32_dtype());
      int ret = check_cuvs(cuvsHnswBuild(resources_, index_params_, &dataset, index_), "build");
      if (OB_FAIL(ret)) { return ret; }
      copy_rows(vectors, ids, extra_infos, size);
      built_ = true;
      return OB_SUCCESS;
    }
    cuvsHnswExtendParams_t params = nullptr;
    int ret = check_cuvs(cuvsHnswExtendParamsCreate(&params), "extend params");
    if (OB_SUCC(ret)) {
      params->num_threads = 0;
      int64_t shape[2] = {size, dim};
      DLManagedTensor dataset;
      init_tensor(dataset, vectors, shape, 2, float32_dtype());
      ret = check_cuvs(cuvsHnswExtend(resources_, params, &dataset, index_), "extend");
    }
    if (params != nullptr) { cuvsHnswExtendParamsDestroy(params); }
    if (OB_SUCC(ret)) { copy_rows(vectors, ids, extra_infos, size); }
    return ret;
  }

  int search(float *query, int dim, int64_t topk, const float *&dist,
             const int64_t *&ids, int64_t &result_size, int ef_search,
             bool need_extra_info, const char *&extra_infos,
             obvsag::FilterInterface *filter, bool reverse_filter,
             bool use_extra_info_filter, float valid_ratio, float distance_threshold)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    dist = nullptr;
    ids = nullptr;
    extra_infos = nullptr;
    result_size = 0;
    if (query == nullptr || dim != dim_ || topk <= 0) { return OB_INVALID_ARGUMENT; }
    if (!built_ || ids_.empty()) { return OB_SUCCESS; }
    const int64_t count = static_cast<int64_t>(ids_.size());
    int64_t candidate = std::max<int64_t>(topk, ef_search > 0 ? ef_search : ef_search_);
    if (valid_ratio > 0.0f && valid_ratio < 1.0f) {
      candidate = std::max<int64_t>(candidate, static_cast<int64_t>(std::ceil(topk / std::max(valid_ratio, 0.05f))));
    }
    candidate = std::min<int64_t>(count, candidate);
    search_params_->ef = static_cast<int32_t>(std::min<int64_t>(candidate, INT32_MAX));
    std::vector<uint64_t> neighbors(static_cast<size_t>(candidate));
    std::vector<float> distances(static_cast<size_t>(candidate));
    int64_t qshape[2] = {1, dim};
    int64_t oshape[2] = {1, candidate};
    DLManagedTensor queries, neighbor_tensor, distance_tensor;
    init_tensor(queries, query, qshape, 2, float32_dtype());
    init_tensor(neighbor_tensor, neighbors.data(), oshape, 2, uint64_dtype());
    init_tensor(distance_tensor, distances.data(), oshape, 2, float32_dtype());
    int ret = check_cuvs(cuvsHnswSearch(resources_, search_params_, index_, &queries,
                                         &neighbor_tensor, &distance_tensor), "search");
    if (OB_FAIL(ret)) { return ret; }
    result_ids_.clear();
    result_distances_.clear();
    result_extra_infos_.clear();
    for (int64_t i = 0; i < candidate && static_cast<size_t>(neighbors[i]) < ids_.size(); ++i) {
      size_t row = static_cast<size_t>(neighbors[i]);
      bool keep = true;
      if (filter != nullptr) {
        bool blocked = use_extra_info_filter && extra_info_size_ > 0
          ? filter->test(extra_infos_.data() + row * extra_info_size_)
          : filter->test(ids_[row]);
        keep = reverse_filter ? blocked : !blocked;
      }
      if (distance_threshold != FLT_MAX && distances[i] > distance_threshold) { keep = false; }
      if (keep) {
        result_ids_.push_back(ids_[row]);
        result_distances_.push_back(distances[i]);
        if (need_extra_info && extra_info_size_ > 0) {
          result_extra_infos_.insert(result_extra_infos_.end(),
                                     extra_infos_.begin() + row * extra_info_size_,
                                     extra_infos_.begin() + (row + 1) * extra_info_size_);
        }
        if (static_cast<int64_t>(result_ids_.size()) >= topk) { break; }
      }
    }
    result_size = static_cast<int64_t>(result_ids_.size());
    dist = result_distances_.empty() ? nullptr : result_distances_.data();
    ids = result_ids_.empty() ? nullptr : result_ids_.data();
    extra_infos = result_extra_infos_.empty() ? nullptr : result_extra_infos_.data();
    return OB_SUCCESS;
  }

  int distance(const float *query, const int64_t *ids, int64_t count, const float *&distances)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (query == nullptr || ids == nullptr || count < 0) { return OB_INVALID_ARGUMENT; }
    distance_results_.resize(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
      auto it = id_to_row_.find(ids[i]);
      if (it == id_to_row_.end()) {
        distance_results_[i] = std::numeric_limits<float>::max();
        continue;
      }
      const float *base = vectors_.data() + it->second * dim_;
      float dot = 0.0f, qnorm = 0.0f, bnorm = 0.0f, l2 = 0.0f;
      for (int j = 0; j < dim_; ++j) {
        float q = query[j], b = base[j];
        dot += q * b; qnorm += q * q; bnorm += b * b; l2 += (q - b) * (q - b);
      }
      if (metric_ == L2Expanded) { distance_results_[i] = l2; }
      else if (metric_ == CosineExpanded) {
        distance_results_[i] = (qnorm == 0.0f || bnorm == 0.0f) ? 1.0f : 1.0f - dot / std::sqrt(qnorm * bnorm);
      } else { distance_results_[i] = -dot; }
    }
    distances = distance_results_.data();
    return OB_SUCCESS;
  }

  int extra(const int64_t *ids, int64_t count, char *out)
  {
    if (ids == nullptr || out == nullptr || count < 0) { return OB_INVALID_ARGUMENT; }
    std::lock_guard<std::mutex> guard(mutex_);
    if (extra_info_size_ == 0) { return OB_SUCCESS; }
    for (int64_t i = 0; i < count; ++i) {
      auto it = id_to_row_.find(ids[i]);
      if (it == id_to_row_.end()) { memset(out + i * extra_info_size_, 0, extra_info_size_); }
      else { memcpy(out + i * extra_info_size_, extra_infos_.data() + it->second * extra_info_size_, extra_info_size_); }
    }
    return OB_SUCCESS;
  }

  int serialize(std::ostream &out)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!built_) { return OB_OP_NOT_ALLOW; }
    TempFile file;
    if (!file.valid()) { return OB_ERR_UNEXPECTED; }
    int ret = check_cuvs(cuvsHnswSerialize(resources_, file.path(), index_), "serialize");
    if (OB_FAIL(ret)) { return ret; }
    std::ifstream in(file.path(), std::ios::binary | std::ios::ate);
    if (!in.good()) { return OB_ERR_UNEXPECTED; }
    uint64_t blob_size = static_cast<uint64_t>(in.tellg());
    in.seekg(0);
    const uint32_t magic = 0x53444355U, version = 1;
    const uint32_t type = static_cast<uint32_t>(index_type_);
    const uint32_t metric = static_cast<uint32_t>(metric_);
    const uint64_t count = ids_.size(), extra = extra_info_size_;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&type), sizeof(type));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&extra), sizeof(extra));
    out.write(reinterpret_cast<const char*>(&blob_size), sizeof(blob_size));
    std::vector<char> buf(1 << 20);
    while (in.good()) { in.read(buf.data(), buf.size()); std::streamsize n = in.gcount(); if (n > 0) out.write(buf.data(), n); }
    if (!ids_.empty()) out.write(reinterpret_cast<const char*>(ids_.data()), sizeof(int64_t) * ids_.size());
    if (!vectors_.empty()) out.write(reinterpret_cast<const char*>(vectors_.data()), sizeof(float) * vectors_.size());
    if (!extra_infos_.empty()) out.write(extra_infos_.data(), extra_infos_.size());
    return out.good() ? OB_SUCCESS : OB_ERR_UNEXPECTED;
  }

  int deserialize(std::istream &in)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    uint32_t magic = 0, version = 0, type = 0, metric = 0;
    int dim = 0; uint64_t count = 0, extra = 0, blob_size = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&type), sizeof(type));
    in.read(reinterpret_cast<char*>(&metric), sizeof(metric));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    in.read(reinterpret_cast<char*>(&extra), sizeof(extra));
    in.read(reinterpret_cast<char*>(&blob_size), sizeof(blob_size));
    if (!in.good() || magic != 0x53444355U || version != 1 || dim != dim_ || extra != extra_info_size_ || type != static_cast<uint32_t>(index_type_) || metric != static_cast<uint32_t>(metric_)) {
      return OB_INVALID_ARGUMENT;
    }
    TempFile file;
    if (!file.valid()) { return OB_ERR_UNEXPECTED; }
    std::ofstream out(file.path(), std::ios::binary | std::ios::trunc);
    std::vector<char> buf(1 << 20);
    uint64_t left = blob_size;
    while (left > 0) { std::streamsize want = static_cast<std::streamsize>(std::min<uint64_t>(left, buf.size())); in.read(buf.data(), want); if (in.gcount() != want) return OB_ERR_UNEXPECTED; out.write(buf.data(), want); left -= static_cast<uint64_t>(want); }
    out.close();
    int ret = check_cuvs(cuvsHnswDeserialize(resources_, index_params_, file.path(), dim_, metric_, index_), "deserialize");
    if (OB_FAIL(ret)) { return ret; }
    ids_.resize(static_cast<size_t>(count));
    vectors_.resize(static_cast<size_t>(count) * dim_);
    extra_infos_.resize(static_cast<size_t>(count) * extra_info_size_);
    if (count > 0) in.read(reinterpret_cast<char*>(ids_.data()), sizeof(int64_t) * count);
    if (count > 0) in.read(reinterpret_cast<char*>(vectors_.data()), sizeof(float) * count * dim_);
    if (!extra_infos_.empty()) in.read(extra_infos_.data(), extra_infos_.size());
    if (!in.good()) { return OB_ERR_UNEXPECTED; }
    id_to_row_.clear();
    for (size_t i = 0; i < ids_.size(); ++i) { id_to_row_[ids_[i]] = i; }
    built_ = true;
    return OB_SUCCESS;
  }

  int number(int64_t &size) const { std::lock_guard<std::mutex> guard(mutex_); size = ids_.size(); return OB_SUCCESS; }
  int type() const { return static_cast<int>(index_type_); }
  int bounds(int64_t &min_id, int64_t &max_id) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (ids_.empty()) { min_id = max_id = 0; return OB_SUCCESS; }
    auto mm = std::minmax_element(ids_.begin(), ids_.end()); min_id = *mm.first; max_id = *mm.second; return OB_SUCCESS;
  }
  uint64_t memory(uint64_t row_count) const { return row_count * (static_cast<uint64_t>(dim_) * sizeof(float) + sizeof(int64_t) + extra_info_size_ + static_cast<uint64_t>(max_degree_) * sizeof(uint64_t)); }
  int optimize() { immutable_ = true; return OB_SUCCESS; }

private:
  void copy_rows(const float *vectors, const int64_t *ids, const char *extra_infos, int size)
  {
    size_t old = ids_.size();
    ids_.insert(ids_.end(), ids, ids + size);
    vectors_.insert(vectors_.end(), vectors, vectors + static_cast<size_t>(size) * dim_);
    if (extra_info_size_ > 0) {
      if (extra_infos != nullptr) extra_infos_.insert(extra_infos_.end(), extra_infos, extra_infos + static_cast<size_t>(size) * extra_info_size_);
      else extra_infos_.resize(extra_infos_.size() + static_cast<size_t>(size) * extra_info_size_, 0);
    }
    for (size_t i = old; i < ids_.size(); ++i) { id_to_row_[ids_[i]] = i; }
  }
  void cleanup()
  {
    if (search_params_ != nullptr) { cuvsHnswSearchParamsDestroy(search_params_); search_params_ = nullptr; }
    if (index_ != nullptr) { cuvsHnswIndexDestroy(index_); index_ = nullptr; }
    if (index_params_ != nullptr) { cuvsHnswIndexParamsDestroy(index_params_); index_params_ = nullptr; }
    if (ace_params_ != nullptr) { cuvsHnswAceParamsDestroy(ace_params_); ace_params_ = nullptr; }
    if (resources_ != 0) { cuvsResourcesDestroy(resources_); resources_ = 0; }
  }
  uint64_t magic_;
  bool built_, immutable_;
  int dim_, max_degree_, ef_construction_, ef_search_;
  obvsag::IndexType index_type_;
  int extra_info_size_;
  cuvsDistanceType metric_;
  cuvsResources_t resources_;
  cuvsHnswIndex_t index_;
  cuvsHnswIndexParams_t index_params_;
  cuvsHnswAceParams_t ace_params_;
  cuvsHnswSearchParams_t search_params_;
  std::vector<int64_t> ids_;
  std::vector<float> vectors_;
  std::vector<char> extra_infos_;
  std::unordered_map<int64_t, size_t> id_to_row_;
  std::vector<float> result_distances_;
  std::vector<int64_t> result_ids_;
  std::vector<char> result_extra_infos_;
  std::vector<float> distance_results_;
  mutable std::mutex mutex_;
};

static CuvsIndexHandler *handler(obvsag::VectorIndexPtr ptr)
{
  return static_cast<CuvsIndexHandler *>(ptr);
}

} // namespace

bool is_index(obvsag::VectorIndexPtr index_handler)
{
  return index_handler != nullptr && *static_cast<const uint64_t *>(index_handler) == CUVS_HANDLER_MAGIC;
}

int validate_create_index(const obvsag::CreateIndexParam &param, std::string &err_msg)
{
  err_msg.clear();
  if (param.is_sparse_) { err_msg = "cuVS supports dense float32 indexes only"; return OB_NOT_SUPPORTED; }
  if (param.index_type_ != obvsag::HNSW_TYPE && param.index_type_ != obvsag::HGRAPH_TYPE) { err_msg = "cuVS supports HNSW indexes only"; return OB_NOT_SUPPORTED; }
  if (param.dtype_ == nullptr || strcasecmp(param.dtype_, "float32") != 0) { err_msg = "cuVS requires float32 vectors"; return OB_NOT_SUPPORTED; }
  if (param.dim_ <= 0 || param.dim_ > 4096) { err_msg = "invalid vector dimension for cuVS"; return OB_INVALID_ARGUMENT; }
  if (param.max_degree_ < 5 || param.ef_construction_ <= param.max_degree_ || param.ef_search_ <= 0) { err_msg = "invalid HNSW parameters for cuVS"; return OB_INVALID_ARGUMENT; }
  cuvsDistanceType metric;
  if (!parse_metric(param.metric_, metric)) { err_msg = "unsupported metric for cuVS"; return OB_NOT_SUPPORTED; }
  return OB_SUCCESS;
}

int create_index(obvsag::VectorIndexPtr &index_handler, obvsag::IndexType index_type,
                 const char *dtype, const char *metric, int dim, int max_degree,
                 int ef_construction, int ef_search, void *, int extra_info_size)
{
  obvsag::CreateIndexParam param;
  param.index_type_ = index_type; param.dtype_ = dtype; param.metric_ = metric; param.dim_ = dim;
  param.max_degree_ = max_degree; param.ef_construction_ = ef_construction; param.ef_search_ = ef_search;
  param.extra_info_size_ = extra_info_size; param.backend_ = 2;
  std::string err;
  int ret = obcuvs::validate_create_index(param, err);
  if (OB_FAIL(ret)) { LOG_WARN("invalid cuVS index parameters", K(ret), KCSTRING(err.c_str())); return ret; }
  int create_ret = OB_SUCCESS;
  CuvsIndexHandler *index = new CuvsIndexHandler(create_ret, index_type, dtype, metric, dim, max_degree, ef_construction, ef_search, extra_info_size);
  if (index == nullptr) { return OB_ALLOCATE_MEMORY_FAILED; }
  if (OB_FAIL(create_ret)) { delete index; return create_ret; }
  index_handler = static_cast<obvsag::VectorIndexPtr>(index);
  return OB_SUCCESS;
}

int create_index(obvsag::VectorIndexPtr &, obvsag::IndexType, const char *, const char *, bool, float, int, void *, int)
{ return OB_NOT_SUPPORTED; }
int build_index(obvsag::VectorIndexPtr ptr, float *v, int64_t *ids, int dim, int size, char *extra) { return is_index(ptr) ? handler(ptr)->build(v, ids, dim, size, extra) : OB_INVALID_ARGUMENT; }
int build_index(obvsag::VectorIndexPtr &, uint32_t *, uint32_t *, float *, int64_t *, int, char *) { return OB_NOT_SUPPORTED; }
int add_index(obvsag::VectorIndexPtr ptr, float *v, int64_t *ids, int dim, int size, char *extra) { return is_index(ptr) ? handler(ptr)->add(v, ids, dim, size, extra) : OB_INVALID_ARGUMENT; }
int add_index(obvsag::VectorIndexPtr &, uint32_t *, uint32_t *, float *, int64_t *, int, char *) { return OB_NOT_SUPPORTED; }
int get_index_number(obvsag::VectorIndexPtr ptr, int64_t &size) { return is_index(ptr) ? handler(ptr)->number(size) : OB_INVALID_ARGUMENT; }
int get_index_type(obvsag::VectorIndexPtr ptr) { return is_index(ptr) ? handler(ptr)->type() : OB_INVALID_ARGUMENT; }
int cal_distance_by_id(obvsag::VectorIndexPtr ptr, const float *v, const int64_t *ids, int64_t count, const float *&dist) { return is_index(ptr) ? handler(ptr)->distance(v, ids, count, dist) : OB_INVALID_ARGUMENT; }
int cal_distance_by_id(obvsag::VectorIndexPtr, uint32_t, uint32_t *, float *, const int64_t *, int64_t, const float *&) { return OB_NOT_SUPPORTED; }
int get_vid_bound(obvsag::VectorIndexPtr ptr, int64_t &min_id, int64_t &max_id) { return is_index(ptr) ? handler(ptr)->bounds(min_id, max_id) : OB_INVALID_ARGUMENT; }
int get_extra_info_by_ids(obvsag::VectorIndexPtr &ptr, const int64_t *ids, int64_t count, char *out) { return is_index(ptr) ? handler(ptr)->extra(ids, count, out) : OB_INVALID_ARGUMENT; }
int knn_search(obvsag::VectorIndexPtr ptr, float *query, int dim, int64_t topk, const float *&dist, const int64_t *&ids, int64_t &size, int ef, bool need, const char *&extra, void *invalid, bool reverse, bool extra_filter, float ratio, void *, float threshold)
{ return is_index(ptr) ? handler(ptr)->search(query, dim, topk, dist, ids, size, ef, need, extra, static_cast<obvsag::FilterInterface *>(invalid), reverse, extra_filter, ratio, threshold) : OB_INVALID_ARGUMENT; }
int knn_search(obvsag::VectorIndexPtr ptr, float *query, int dim, int64_t topk, const float *&dist, const int64_t *&ids, int64_t &size, int ef, bool need, const char *&extra, void *invalid, bool reverse, bool extra_filter, float ratio, void *&iter, bool last, void *alloc)
{ iter = nullptr; return knn_search(ptr, query, dim, topk, dist, ids, size, ef, need, extra, invalid, reverse, extra_filter, ratio, alloc, FLT_MAX); }
int knn_search(obvsag::VectorIndexPtr, uint32_t, uint32_t *, float *, int64_t, const float *&, const int64_t *&, const char *&, int64_t &, float, int64_t, void *, bool, bool, float, void *, bool)
{ return OB_NOT_SUPPORTED; }
int fserialize(obvsag::VectorIndexPtr ptr, std::ostream &out) { return is_index(ptr) ? handler(ptr)->serialize(out) : OB_INVALID_ARGUMENT; }
int fdeserialize(obvsag::VectorIndexPtr &ptr, std::istream &in) { return is_index(ptr) ? handler(ptr)->deserialize(in) : OB_INVALID_ARGUMENT; }
int delete_index(obvsag::VectorIndexPtr &ptr) { if (!is_index(ptr)) return OB_INVALID_ARGUMENT; delete handler(ptr); ptr = nullptr; return OB_SUCCESS; }
uint64_t estimate_memory(obvsag::VectorIndexPtr ptr, uint64_t rows, bool) { return is_index(ptr) ? handler(ptr)->memory(rows) : 0; }
int immutable_optimize(obvsag::VectorIndexPtr &ptr) { return is_index(ptr) ? handler(ptr)->optimize() : OB_INVALID_ARGUMENT; }

} // namespace obcuvs
} // namespace common
} // namespace oceanbase
#endif
