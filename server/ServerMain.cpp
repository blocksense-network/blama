// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>
#include <llama/ControlVector.hpp>

#include <jalog/Instance.hpp>
#include <jalog/sinks/DefaultSink.hpp>

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include <iostream>

#include "ac-test-data-llama-dir.h"

namespace net = boost::asio;
using tcp = net::ip::tcp;
namespace beast = boost::beast;
namespace http = beast::http;

class thread_runner {
    std::vector<std::thread> m_threads; // would use jthread, but apple clang still doesn't support them
public:
    thread_runner() = default;

    template <typename Ctx>
    void start(Ctx& ctx, size_t n) {
        assert(m_threads.empty());
        if (!m_threads.empty()) return; // rescue
        m_threads.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            m_threads.push_back(std::thread([i, n, &ctx]() mutable {
                ctx.run();
            }));
        }
    }

    void join() {
        for (auto& t : m_threads) {
            t.join();
        }
        m_threads.clear();
    }

    template <typename Ctx>
    thread_runner(Ctx& ctx, size_t n) {
        start(ctx, n);
    }

    ~thread_runner() {
        join();
    }

    size_t num_threads() const noexcept {
        return m_threads.size();
    }

    bool empty() const noexcept {
        return m_threads.empty();
    }
};

class Server {

    bl::llama::Model m_model;

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
public:

    Server(const std::string& modelGguf) :
        m_model(modelGguf, {}, modelLoadProgressCallback)
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
            // TODO: instance pool
            // free instances can actually be stored in an ac-io channel, take one, then put it back
            // the channel itself gives us all the async ops (await, cb) to get and put an instance
            bl::llama::Instance instance(m_model, {});

            auto& session = instance.startSession({});

            auto& vocab = m_model.vocab();
            session.setInitialPrompt(vocab.tokenize(req.body(), true, true));

            std::stringstream ss;
            for (int i = 0; i < 200; ++i) {
                auto pred = session.getToken();
                if (pred.token == bl::llama::Token_Invalid) {
                    // no more tokens
                    break;
                }
                ss << vocab.tokenToString(pred.token);
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
            net::co_spawn(ex, handleRequest(beast::tcp_stream(std::move(sock))), net::detached);
        }
    }

};

int main() {
    jalog::Instance jl;
    jl.setup().async().add<jalog::sinks::DefaultSink>();

    bl::llama::initLibrary();

    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";

    Server server(modelGguf);

    net::io_context ioctx;
    auto guard = net::make_work_guard(ioctx);

    thread_runner runner(ioctx, 4);

    net::co_spawn(ioctx, server.listen(), net::detached);
}
