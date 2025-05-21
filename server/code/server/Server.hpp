// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include <memory>
#include <string>
#include <vector>
#include <itlib/ufunction.hpp>

namespace bl::llama {

class Model;

namespace server {

class BL_LLAMA_SERVER_API Server {
public:
    Server(std::shared_ptr<Model> model);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    struct CompleteRequestParams {
        std::string prompt;
        uint32_t maxTokens = 0;
        uint32_t seed = 0;
        std::string suffix;
        float temperature = 0.7f;
    };

    struct TokenData {
        std::string_view tokenStr;
        uint32_t tokenId = 0;
        struct LogitData {
            uint32_t tokenId = 0;
            float logit = 0;
        };
        std::vector<LogitData> logits;
    };

    void completeText(CompleteRequestParams params, itlib::ufunction<void(std::vector<TokenData>)> cb);

    void verify(CompleteRequestParams req, std::vector<TokenData> resp, itlib::ufunction<void(float)> cb);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace server
} // namespace bl::llama
