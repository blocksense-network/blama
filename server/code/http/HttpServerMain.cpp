// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include <cstring>
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>
#include <llama/ControlVector.hpp>

#include <server/Server.hpp>

#include <jalog/Instance.hpp>
#include <jalog/sinks/DefaultSink.hpp>
#include <jalog/Log.hpp>

#include <bstl/thread_runner.hpp>
#include <bstl/iile.h>
#include <bstl/move.hpp>

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <concepts>

#include "ac-test-data-llama-dir.h"

namespace net = boost::asio;
using tcp = net::ip::tcp;
namespace beast = boost::beast;
namespace http = beast::http;
namespace fs = std::filesystem;

nlohmann::json toJson(bl::llama::server::Server::CompleteReponse& gen) {
    auto jsonTokens = nlohmann::json::array();
    for (auto& g : gen) {
        auto& jt = jsonTokens.emplace_back();
        jt["str"] = std::move(g.tokenStr);
        jt["id"] = g.tokenId;
        auto& jlg = jt["logits"] = nlohmann::json::array();
        for (auto& l : g.logits) {
            auto& jl = jlg.emplace_back();
            jl["id"] = l.tokenId;
            jl["logit"] = l.logit;
        }
    }
    return jsonTokens;
}

bl::llama::server::Server::CompleteReponse toCompleteResponse(nlohmann::json& json) {
    bl::llama::server::Server::CompleteReponse gen;
    auto& jsonTokens = json["tokenData"];
    gen.reserve(jsonTokens.size());
    for (auto& jt : jsonTokens) {
        auto& g = gen.emplace_back();
        g.tokenStr = std::move(jt["str"].get_ref<std::string&>());
        g.tokenId = jt["id"].get<int>();
        auto& jlg = jt["logits"];
        g.logits.reserve(jlg.size());
        for (auto& jl : jlg) {
            auto& l = g.logits.emplace_back();
            l.tokenId = jl["id"].get<int>();
            l.logit = jl["logit"].get<float>();
        }
    }
    return gen;
}

template <typename T>
void opt_get(nlohmann::json& dict, std::string_view key, T& value) {
    auto it = dict.find(key);
    if (it != dict.end()) {
        if constexpr (std::same_as<T, std::string>) {
            value = it->get_ref<std::string&>();
        }
        else {
            value = it->get<T>();
        }
    }
}

bl::llama::server::Server::CompleteRequestParams toCompleteParams(nlohmann::json& json) {
    bl::llama::server::Server::CompleteRequestParams params;
    params.prompt = json["prompt"].get_ref<std::string&>();
    opt_get(json, "max_tokens", params.maxTokens);
    opt_get(json, "seed", params.seed);
    opt_get(json, "suffix", params.suffix);
    opt_get(json, "temp", params.temperature);
    opt_get(json, "top_p", params.topP);
    return params;
}

bl::llama::server::Server::ChatCompleteRequestParams toChatCompleteParams(nlohmann::json& json) {
    bl::llama::server::Server::ChatCompleteRequestParams params;
    if (json.contains("messages") && json["messages"].is_array()) {
        params.messages.reserve(json["messages"].size());
        for (const auto& messageJson : json["messages"]) {
            bl::llama::server::Server::ChatCompleteRequestParams::Message msg;
            // Assuming Message has role and content fields
            if (messageJson.contains("role")) {
                msg.role = messageJson["role"].get<std::string>();
            }
            if (messageJson.contains("content")) {
                msg.content = messageJson["content"].get<std::string>();
            }
            params.messages.push_back(std::move(msg));
        }
    }
    opt_get(json, "max_tokens", params.maxTokens);
    opt_get(json, "seed", params.seed);
    opt_get(json, "temp", params.temperature);
    opt_get(json, "top_p", params.topP);
    return params;
}

bl::llama::server::Server::CompleteRequestParams toCompleteParams(std::string_view jsonStr) {
    auto json = nlohmann::json::parse(jsonStr);
    return toCompleteParams(json);
}

bl::llama::server::Server::ChatCompleteRequestParams toChatCompleteParams(std::string_view jsonStr) {
    auto json = nlohmann::json::parse(jsonStr);
    return toChatCompleteParams(json);
}

class Server {
    std::shared_ptr<bl::llama::Model> m_model;
    bl::llama::server::Server m_server;

    static bool modelLoadProgressCallback(float progress) {
        static bool initialized = false;
        const int barWidth = 50;

        // Clamp progress
        progress = std::max(0.0f, std::min(1.0f, progress));

        if (!initialized) {
            std::cout << "Loading: ";
            initialized = true;
        }

        // Clear current line and move cursor to start
        std::cout << '\r';

        // Build progress bar
        std::cout << "Loading: [";
        int filled = static_cast<int>(progress * barWidth);

        for (int i = 0; i < barWidth; i++) {
            if (i < filled) {
                std::cout << '=';
            } else if (i == filled && progress < 1.0f) {
                std::cout << '>';
            } else {
                std::cout << ' ';
            }
        }

        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << '%';
        std::cout.flush();

        if (progress >= 1.0f) {
            std::cout << " - Complete!\n";
            initialized = false; // Reset for next use
        }

        return true;
    }

    template <typename T>
    struct AsyncCompleteOp {
        net::any_io_executor ex;
        bl::llama::server::Server& server;
        T params;

        template <typename Self>
        void operator()(Self& self) {
            auto takeParams = bstl::move(params);
            if constexpr (std::is_same_v<T, bl::llama::server::Server::CompleteRequestParams>) {
                server.completeText(bstl::move(takeParams), [ex = bstl::move(ex), self = bstl::move(self)](bl::llama::server::Server::CompleteReponse gen) mutable {
                    post(ex, [self = bstl::move(self), gen = bstl::move(gen)]() mutable {
                        self.complete(bstl::move(gen));
                    });
                });
            } else if constexpr (std::is_same_v<T, bl::llama::server::Server::ChatCompleteRequestParams>) {
                server.chatComplete(bstl::move(takeParams), [ex = bstl::move(ex), self = bstl::move(self)](bl::llama::server::Server::CompleteReponse gen) mutable {
                    post(ex, [self = bstl::move(self), gen = bstl::move(gen)]() mutable {
                        self.complete(bstl::move(gen));
                    });
                });
            } else {
                static_assert(false, "Unsupported parameter type for AsyncCompleteOp");
            }
        }
    };

    decltype(auto) asyncComplete(net::any_io_executor ex, bl::llama::server::Server::CompleteRequestParams params) {
        return net::async_compose<const net::use_awaitable_t<>, void(bl::llama::server::Server::CompleteReponse)>(
            AsyncCompleteOp<bl::llama::server::Server::CompleteRequestParams>{.ex = ex, .server = m_server, .params = std::move(params)}, net::use_awaitable, ex
        );
    }

    decltype(auto) asyncChatComplete(net::any_io_executor ex, bl::llama::server::Server::ChatCompleteRequestParams params) {
        return net::async_compose<const net::use_awaitable_t<>, void(bl::llama::server::Server::CompleteReponse)>(
            AsyncCompleteOp<bl::llama::server::Server::ChatCompleteRequestParams>{.ex = ex, .server = m_server, .params = std::move(params)}, net::use_awaitable, ex
        );
    }

    template <typename T>
    struct AsyncVerifyOp {
        net::any_io_executor ex;
        bl::llama::server::Server& server;
        T params;
        bl::llama::server::Server::CompleteReponse response;

        template <typename Self>
        void operator()(Self& self) {
            auto takeParams = bstl::move(params);
            auto takeResponse = bstl::move(response);

            if constexpr (std::is_same_v<T, bl::llama::server::Server::CompleteRequestParams>) {
                server.verify(bstl::move(takeParams), bstl::move(takeResponse), [ex = bstl::move(ex), self = bstl::move(self)](float result) mutable {
                    post(ex, [self = bstl::move(self), result]() mutable {
                        self.complete(result);
                    });
                });
            } else if constexpr (std::is_same_v<T, bl::llama::server::Server::ChatCompleteRequestParams>) {
                server.chatVerify(bstl::move(takeParams), bstl::move(takeResponse), [ex = bstl::move(ex), self = bstl::move(self)](float result) mutable {
                    post(ex, [self = bstl::move(self), result]() mutable {
                        self.complete(result);
                    });
                });
            } else {
                static_assert(false, "Unsupported parameter type for AsyncCompleteOp");
            }
        }
    };

    decltype(auto) asyncVerify(net::any_io_executor ex, bl::llama::server::Server::CompleteRequestParams params, bl::llama::server::Server::CompleteReponse response) {
        return net::async_compose<const net::use_awaitable_t<>, void(float)>(
            AsyncVerifyOp<bl::llama::server::Server::CompleteRequestParams>{.ex = ex, .server = m_server, .params = std::move(params), .response = std::move(response)}, net::use_awaitable, ex
        );
    }

    decltype(auto) asyncChatVerify(net::any_io_executor ex, bl::llama::server::Server::ChatCompleteRequestParams params, bl::llama::server::Server::CompleteReponse response) {
        return net::async_compose<const net::use_awaitable_t<>, void(float)>(
            AsyncVerifyOp<bl::llama::server::Server::ChatCompleteRequestParams>{.ex = ex, .server = m_server, .params = std::move(params), .response = std::move(response)}, net::use_awaitable, ex
        );
    }

    template <typename T>
    decltype(auto) getCompleteResponse(T& gen, const http::request<http::string_body>& req) {
        std::ostringstream ss;
        for (auto& g : gen) {
            ss << g.tokenStr;
        }

        nlohmann::json outJson;
        outJson["text"] = ss.str();
        outJson["tokenData"] = toJson(gen);

        // Prepare the response
        http::response<http::string_body> res(http::status::ok, req.version());
        res.set(http::field::server, "Beast");
        res.set(http::field::content_type, "text/json");
        res.set(http::field::access_control_allow_origin, "*");
        res.keep_alive(req.keep_alive());
        res.body() = outJson.dump();
        res.prepare_payload();

        return res;
    }

    template <typename T>
    decltype(auto) getVerifyResponse(T& verifyResult, const http::request<http::string_body>& req) {
        http::response<http::string_body> res(http::status::ok, req.version());
        res.set(http::field::server, "Beast");
        res.set(http::field::content_type, "text/json");
        res.set(http::field::access_control_allow_origin, "*");
        res.keep_alive(req.keep_alive());
        res.body() = nlohmann::json({{"result", verifyResult}}).dump();
        res.prepare_payload();

        return res;
    }


public:

    Server(const std::string& modelGguf)
        : m_model(std::make_shared<bl::llama::Model>(modelGguf, bl::llama::Model::Params{}, modelLoadProgressCallback))
        , m_server(m_model)
    {}

    net::awaitable<void> handleRequest(beast::tcp_stream stream) {
        beast::flat_buffer buffer;
        http::request<http::string_body> req;

        auto ex = co_await net::this_coro::executor;

        co_await http::async_read(stream, buffer, req, net::use_awaitable);

        if (req.method() != http::verb::post) {
            http::response<http::empty_body> res(http::status::bad_request, req.version());
            res.set(http::field::access_control_allow_origin, "*");
            co_await http::async_write(stream, res, net::use_awaitable);
        }
        else if (req.target() == "/complete") {
            auto params = toCompleteParams(req.body());

            auto gen = co_await asyncComplete(ex, std::move(params));
            auto res = getCompleteResponse(gen, req);
            // Write the response
            co_await http::async_write(stream, res, net::use_awaitable);
        }
        else if(req.target() == "/chat/completions") {
            auto params = toChatCompleteParams(req.body());

            auto gen = co_await asyncChatComplete(ex, std::move(params));
            auto res = getCompleteResponse(gen, req);

            // Write the response
            co_await http::async_write(stream, res, net::use_awaitable);
        }
        else if (req.target() == "/verify_completion") {
            auto json = nlohmann::json::parse(req.body());
            auto rreq = toCompleteParams(json["request"]);
            auto rrsp = toCompleteResponse(json["response"]);

            auto verifyResult = co_await asyncVerify(ex, std::move(rreq), std::move(rrsp));
            auto res = getVerifyResponse(verifyResult, req);

            // Write the response
            co_await http::async_write(stream, res, net::use_awaitable);
        }
        else if (req.target() == "/chat/verify_completion") {
            auto json = nlohmann::json::parse(req.body());
            auto rreq = toChatCompleteParams(json["request"]);
            auto rrsp = toCompleteResponse(json["response"]);

            auto verifyResult = co_await asyncChatVerify(ex, std::move(rreq), std::move(rrsp));
            auto res = getVerifyResponse(verifyResult, req);

            // Write the response
            co_await http::async_write(stream, res, net::use_awaitable);
        }
        else {
            http::response<http::empty_body> res(http::status::not_found, req.version());
            res.set(http::field::access_control_allow_origin, "*");
            co_await http::async_write(stream, res, net::use_awaitable);
        }

        // Close the stream
        stream.socket().shutdown(tcp::socket::shutdown_send);
    }

    net::awaitable<void> listen(const boost::asio::ip::address &addr, net::ip::port_type port) {
        auto ex = co_await net::this_coro::executor;
        tcp::acceptor acc(ex, tcp::endpoint(addr, port));

        while (true) {
            auto sock = co_await acc.async_accept(net::use_awaitable);
            net::co_spawn(ex, handleRequest(beast::tcp_stream(bstl::move(sock))), net::detached);
        }
    }

};

int main(int argc, char* argv[]) {
    jalog::Instance jl;
    jl.setup().async().add<jalog::sinks::DefaultSink>();

    bl::llama::initLibrary();

    // Default values
    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    boost::asio::ip::address host = boost::asio::ip::make_address("0.0.0.0");
    net::ip::port_type port = 7331;

    const char* host_env = std::getenv("BLAMA_HOST");
    if (host_env) {
        try {
            host = boost::asio::ip::make_address(host_env);
        } catch (const boost::exception &e) {
            // TODO: have to link against `Boost::exception` in `../CMakeLists.txt` somehow
            // std::string diag = boost::exception_detail::get_diagnostic_information(
            //     e, "Invalid BLAMA_HOST");
            // throw std::invalid_argument(diag);
            throw std::invalid_argument("Invalid BLAMA_HOST");
        }
    }

    const char* port_env = std::getenv("BLAMA_PORT");
    if (port_env) {
        size_t idx = 0;
        unsigned long value = std::stoul(port_env, &idx, 10);

        if (idx != std::strlen(port_env)) {
            throw std::invalid_argument("Extra characters after BLAMA_PORT number");
        }

        if (value > std::numeric_limits<net::ip::port_type>::max()) {
            throw std::out_of_range("Value exceeds uint16_t max");
        }

        port = static_cast<net::ip::port_type>(value);;
    }

    const char* model_env = std::getenv("BLAMA_MODEL");
    if (model_env) {
        if (!model_env || std::string(model_env).empty()) {
            throw std::runtime_error(std::string("Environment variable not set or empty: ") + model_env);
        }

        std::string modelPathString(model_env);

        if (!modelPathString.ends_with(".gguf")) {
            throw std::runtime_error(std::string("BLAMA_MODEL does not end with .gguf: ") + modelPathString);
        }

        fs::path model_path(modelPathString);

        if (!fs::exists(model_path)) {
            throw std::runtime_error("BLAMA_MODEL does not exist: " + modelPathString);
        }

        if (!fs::is_regular_file(model_path)) {
            throw std::runtime_error("BLAMA_MODEL is not a regular file: " + modelPathString);
        }

        modelGguf = modelPathString;
    }

    JALOG(Info, "Loading model ", modelGguf);
    JALOG(Info, "Listening on port ", port);

    Server server(modelGguf);

    net::io_context ioctx;
    auto guard = net::make_work_guard(ioctx);

    bstl::thread_runner runner(ioctx, 4);

    net::co_spawn(ioctx, server.listen(host, port), net::detached);
}
