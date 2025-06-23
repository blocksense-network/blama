// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
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

    net::awaitable<void> listen(net::ip::port_type port) {
        auto ex = co_await net::this_coro::executor;
        tcp::acceptor acc(ex, tcp::endpoint(tcp::v4(), port));

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

    // --- Configuration with Defaults ---
    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    net::ip::port_type port = 7331; // Default port

    // --- Read Port from Environment Variable (Fallback) ---
    const char* port_env = std::getenv("PORT");
    if (port_env) {
        try {
            port = std::stoi(port_env);
        } catch (const std::invalid_argument& e) {
            JALOG(Error, "Warning: Invalid PORT environment variable '", port_env, "'. Using default port ", port, ".");
        } catch (const std::out_of_range& e) {
            JALOG(Error, "Warning: PORT environment variable '", port_env, "' is out of range. Using default port ", port, ".");
        }
    }

    // Parse `--port` argument (popping it out if found)
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--port") {
            if (i + 1 < args.size()) {
                try {
                    port = std::stoi(args[i + 1]);
                    // Erase the flag and its value so the positional argument logic still works
                    args.erase(args.begin() + i, args.begin() + i + 2);
                    --i; // Decrement i to re-evaluate the new element at the current position
                } catch (const std::invalid_argument& e) {
                    JALOG(Error, "Error: Invalid argument for --port. Please provide a valid number.");
                    return 1;
                } catch (const std::out_of_range& e) {
                    JALOG(Error, "Error: Port number is out of the valid range.");
                    return 1;
                }
            } else {
                JALOG(Error, "Error: --port flag requires an argument.");
                return 1;
            }
        }
    }

    // Handle positional argument for model path
    if (args.size() == 1) {
        modelGguf = args[0];
    } else if (args.size() > 1) {
        std::cerr << "Usage: " << argv[0] << " [--port <port>] [model.gguf]" << std::endl;
        return 1;
    }

    JALOG(Info, "Loading ", modelGguf);
    JALOG(Info, "Listening on port ", port);

    Server server(modelGguf);

    net::io_context ioctx;
    auto guard = net::make_work_guard(ioctx);

    bstl::thread_runner runner(ioctx, 4);

    net::co_spawn(ioctx, server.listen(port), net::detached);
}
