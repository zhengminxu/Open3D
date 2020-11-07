#pragma once

#include "torch/script.h"

int64_t Nms(torch::Tensor boxes, torch::Tensor keep, double nms_overlap_thresh);
