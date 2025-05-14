// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "LogitComparer.hpp"
#include <cmath>

namespace {
std::unordered_map<int32_t, float> softmax(const bl::llama::TokenDataVector& data) {
    std::unordered_map<int32_t, float> result(data.size());

    // Step 1: Find max logit to subtract for numerical stability
    float maxLogit = data[0].logit;

    // Step 2: Compute exp(logit - maxLogit) for each element
    float sumExp = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        float p = std::exp(data[i].logit - maxLogit);
        result[data[i].token] = p;
        sumExp += p;
    }

    // Step 3: Normalize to get probabilities
    for (auto& val : result) {
        val.second /= sumExp;
    }

    return result;
}
}

namespace bl::llama {

// Generate 4 step comparison
// 1. Compare the top1 token
// 2. Compare the euclidean distance of the logits
//  - If the distance is ≤ 2% of the max distance, we can consider them equal
// 3. Compare the Jensen-Shannon divergence of the probabilities
//  - If the divergence is ≤ 0.05, we can consider them equal
LogitComparer::ComparisonMetrics LogitComparer::compare(const TokenDataVector& data1, const TokenDataVector& data2) {
    ComparisonMetrics metrics;
    metrics.top1Match = data1[0].token == data2[0].token ? 1.0f : 0.0f;

    const auto minSize = std::min(data1.size(), data2.size());
    float distance1 = euclidean_distance_sq({data1.data(), minSize});
    float distance2 = euclidean_distance_sq({data2.data(), minSize});

    metrics.distance = std::fabs(distance1 - distance2) / std::max(distance1, distance2);

    auto prob_map = softmax(data1);
    auto prob_map2 = softmax(data2);

    metrics.jsd = jsd(prob_map, prob_map2);

    return metrics;
}

// Final score is a weighted sum of the metrics.
// if the final score is ≥ 0.95 we can consider them equal
float LogitComparer::comparisonFinalScore(std::span<ComparisonMetrics> metrics) {
    double total = 0.0f;
    for (auto& m : metrics) {
        total +=
            0.5 * (1.0f - m.distance) +
            0.5 * (1.0f - m.jsd);
    }
    return float(total / metrics.size());
}

float LogitComparer::logitSimilarity(const TokenDataVector& data1, const TokenDataVector& data2) {
    std::unordered_map<int32_t, float> l_map, l2_map;

    for (const auto& t : data1) l_map[t.token] = t.logit;
    for (const auto& t : data2) l2_map[t.token] = t.logit;

    auto prob1 = softmax(data1);
    auto prob2 = softmax(data2);

    float weightedSimSum = 0.0f;
    float totalWeight = 0.0f;
    for (auto& t : data1) {
        float weight = std::abs(t.logit);
        float sim = 0.0f;
        if (l2_map.count(t.token)) {
            sim = 1 - (std::abs(t.logit - l2_map[t.token]) / std::abs(std::max(t.logit, l2_map[t.token])));
        }

        weightedSimSum += weight * sim;
        totalWeight += weight;
    }

    return totalWeight > 0.0f ? (weightedSimSum / totalWeight) : 0.0f;
}

float LogitComparer::jsd(const std::unordered_map<Token, float>& probs1, const std::unordered_map<Token, float>& probs2) {
    std::unordered_map<Token, float> avg_dist;
    for (const auto& [token, p] : probs1) {
        if (probs2.count(token)) {
            avg_dist[token] = (p + probs2.at(token)) / 2.0f;
        }
    }

    auto kl_divergence = [](const std::unordered_map<Token, float>& P, const std::unordered_map<Token, float>& Q) {
        float kl = 0.0f;
        for (const auto& [token, p] : P) {
            if (p > 0.0f && Q.count(token) && Q.at(token) > 0.0f) {
                kl += p * std::log(p / Q.at(token));
            }
        }
        return kl;
    };

    auto div1 = kl_divergence(probs1, avg_dist);
    auto div2 = kl_divergence(probs2, avg_dist);

    return (div1 + div2) / 2.0f;
}

float LogitComparer::euclidean_distance_sq(std::span<const TokenData> tokens) {
    float distance = 0.0f;
    for (auto& t : tokens) {
        distance += t.logit * t.logit;
    }

    // To achieve total result, we need to take the square root of the sum,
    // but since we don't need it to be accurate, we can skip it
    return distance;
}

}
