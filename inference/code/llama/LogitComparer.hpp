// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "Token.hpp"
#include <unordered_map>
#include <span>

namespace bl::llama {

struct ComparisonMetrics {
    float top1Match;
    float distance;
    float jsd;
};
class BL_LLAMA_API LogitComparer {
public:

    static ComparisonMetrics compare(const TokenDataVector& data1, const TokenDataVector& data2);
    static float logitSimilarity(const TokenDataVector& data1, const TokenDataVector& data2);

private:
    static float jsd(const std::unordered_map<Token, float>& logits1, const std::unordered_map<Token, float>& logits2);
    static float euclidean_distance_sq(std::span<const TokenData> tokens);
};


struct BL_LLAMA_API MetricsAggregator {
    float pushAndVerify(std::span<const ComparisonMetrics> m);

private:
    std::vector<ComparisonMetrics> metrics;
};
}
