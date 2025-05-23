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

class Server {
    std::shared_ptr<bl::llama::Model> m_model;
    bl::llama::server::Server m_server;

    static bool modelLoadProgressCallback(float progress) {
        const int barWidth = 50;
        static float currProgress = 0;
        auto delta = int(progress * barWidth) - int(currProgress * barWidth);
        for (int i = 0; i < delta; i++) {
            std::cout.put('=');
        }
        currProgress = progress;
        if (progress == 1.f) {
            std::cout << '\n';
        }
        return true;
    }

    template <typename T>
    static void opt_get(nlohmann::json& dict, std::string_view key, T& value) {
        auto it = dict.find(key);
        if (it != dict.end()) {
            if constexpr(std::same_as<T, std::string>) {
                value = it->get_ref<std::string&>();
            }
            else {
                value = it->get<T>();
            }
        }
    }

    struct AsyncCompleteOp {
        net::any_io_executor ex;
        bl::llama::server::Server& server;
        bl::llama::server::Server::CompleteRequestParams params;

        template <typename Self>
        void operator()(Self& self) {
            auto takeParams = bstl::move(params);
            server.completeText(bstl::move(takeParams), [ex = bstl::move(ex), self = bstl::move(self)](std::vector<bl::llama::server::Server::TokenData> gen) mutable {
                post(ex, [self = bstl::move(self), gen = bstl::move(gen)]() mutable {
                    self.complete(bstl::move(gen));
                });
            });
        }
    };

    decltype(auto) asyncComplete(net::any_io_executor ex, bl::llama::server::Server::CompleteRequestParams params) {
        return net::async_compose<const net::use_awaitable_t<>, void(std::vector<bl::llama::server::Server::TokenData>)>(
            AsyncCompleteOp{ .ex = ex, .server = m_server, .params = std::move(params) }, net::use_awaitable, ex
        );
    }
public:

    Server(const std::string& modelGguf)
        : m_model(std::make_shared<bl::llama::Model>(modelGguf, bl::llama::Model::Params{}, modelLoadProgressCallback))
        , m_server(m_model)
    {}

    net::awaitable<void> handleRequest(beast::tcp_stream stream) {
        beast::flat_buffer buffer;
        http::request<http::string_body> req;

        co_await http::async_read(stream, buffer, req, net::use_awaitable);

        if (req.target() != "/complete") {
            http::response<http::empty_body> res(http::status::not_found, req.version());
            res.set(http::field::access_control_allow_origin, "*");
            co_await http::async_write(stream, res, net::use_awaitable);
        }
        else if (req.method() != http::verb::post) {
            http::response<http::empty_body> res(http::status::bad_request, req.version());
            res.set(http::field::access_control_allow_origin, "*");
            co_await http::async_write(stream, res, net::use_awaitable);
        }
        else {
            auto params = iile([&]() {
                auto json = nlohmann::json::parse(req.body());

                bl::llama::server::Server::CompleteRequestParams params;
                params.prompt = bstl::move(json["prompt"].get_ref<std::string&>());
                opt_get(json, "max_tokens", params.maxTokens);
                opt_get(json, "seed", params.seed);
                opt_get(json, "suffix", params.suffix);
                opt_get(json, "temp", params.temperature);
                opt_get(json, "top_p", params.topP);
                return params;
            });

            auto ex = co_await net::this_coro::executor;
            auto gen = co_await asyncComplete(ex, std::move(params));

            std::ostringstream ss;
            for (auto& g : gen) {
                ss << g.tokenStr;
            }

            // Prepare the response
            http::response<http::string_body> res(http::status::ok, req.version());
            res.set(http::field::server, "Beast");
            res.set(http::field::content_type, "text/plain");
            res.set(http::field::access_control_allow_origin, "*");
            res.keep_alive(req.keep_alive());
            res.body() = ss.str();
            res.prepare_payload();

            // Write the response
            co_await http::async_write(stream, res, net::use_awaitable);
        }

        // Close the stream
        stream.socket().shutdown(tcp::socket::shutdown_send);
    }


    net::awaitable<void> listen() {
        auto ex = co_await net::this_coro::executor;
        tcp::acceptor acc(ex, tcp::endpoint(tcp::v4(), 7331));

        while (true) {
            auto sock = co_await acc.async_accept(net::use_awaitable);
            net::co_spawn(ex, handleRequest(beast::tcp_stream(bstl::move(sock))), net::detached);
        }
    }

};

int main() {
    jalog::Instance jl;
    jl.setup().async().add<jalog::sinks::DefaultSink>();

    bl::llama::initLibrary();

    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    //std::string modelGguf = "/Users/pacominev/repos/ac/ac-dev/ilib-llama.cpp/tmp/Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf";

    Server server(modelGguf);

    net::io_context ioctx;
    auto guard = net::make_work_guard(ioctx);

    bstl::thread_runner runner(ioctx, 4);

    net::co_spawn(ioctx, server.listen(), net::detached);
}
