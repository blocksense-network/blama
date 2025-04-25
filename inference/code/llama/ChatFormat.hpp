// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "ChatMsg.hpp"

#include <memory>
#include <span>

struct llama_chat_message;

namespace minja {
class chat_template;
}
namespace bl::llama {
class Model;

class BL_LLAMA_API ChatFormat {
public:
    struct Params {
        std::string chatTemplate;
        std::string bosToken;
        std::string eosToken;
    };

    explicit ChatFormat(std::string templateStr);
    explicit ChatFormat(Params params);

    ~ChatFormat();

    static Params getChatParams(const Model& model);

    const std::string& tpl() const noexcept { return m_templateStr; }

    // wrapper around llama_chat_apply_template
    // throw an error on unsupported template
    std::string formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt) const ;

    // format single message taking history into account
    std::string formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt = false) const;

    class impl;
private:
    std::string m_templateStr;
    std::unique_ptr<impl> m_impl;
};

} // namespace bl::llama
