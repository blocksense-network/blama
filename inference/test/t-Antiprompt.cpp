// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <doctest/doctest.h>
#include <vector>

#include "llama/AntipromptManager.hpp"

TEST_CASE("incremental finder - empty") {
    // by default string finder has empty search string
    // and will always return false
    bl::llama::IncrementalStringFinder f("");
    CHECK(f.feedText("") == -1);
    CHECK(f.feedText("empty") == -1);

    f = bl::llama::IncrementalStringFinder("demo");
    // empty feed
    CHECK(f.feedText("") == -1);
}

TEST_CASE("incremental finder - partial match") {
    bl::llama::IncrementalStringFinder f("demo");
    CHECK(f.feedText("de") == -1);
    CHECK(f.feedText("mo") == 2);

    f = bl::llama::IncrementalStringFinder("the");
    // no match
    CHECK(f.feedText("empty") == -1);

    // complex partial match
    CHECK(f.feedText("emptyth") == -1); // last 2 are 'th'
    CHECK(f.feedText("ehooooo") == 1); // + 'e' from the start
}

TEST_CASE("incremental finder - substring") {
    bl::llama::IncrementalStringFinder f("demo");
    // complex substring
    CHECK(f.feedText("dede") == -1); // will find only 2
    CHECK(f.feedText("demo2") == 4); // has the contaning string
}

TEST_CASE("incremental finder - case sensitivity") {
    // case sensitivity
    bl::llama::IncrementalStringFinder f("The");
    CHECK_FALSE(f.feedText("the") == 3);
}

TEST_CASE("antiprompt manager - empty") {
    bl::llama::AntipromptManager am;
    am.addAntiprompt("");
    CHECK(am.feedGeneratedText("empty").empty());

    am.addAntiprompt("user:");
    CHECK(am.feedGeneratedText("").empty());
}

TEST_CASE("antiprompt manager - detect") {
    bl::llama::AntipromptManager am;
    am.addAntiprompt("exit");
    am.addAntiprompt("quit");
    CHECK(am.feedGeneratedText("please continue").empty());
    CHECK(am.feedGeneratedText("please exit!") == "exit!");
    CHECK(am.feedGeneratedText("please quit now!") == "quit now!");
}

TEST_CASE("antiprompt manager - incremental feed") {
    bl::llama::AntipromptManager am;
    am.addAntiprompt("downstream");
    am.addAntiprompt("shutdown");

    CHECK(am.feedGeneratedText("shut").empty());          // Partial match, so false
    CHECK(am.feedGeneratedText("down") == "shutdown");    // Completes the match, so true

    CHECK(am.feedGeneratedText("stream").empty()); // state should be reset after match
}

TEST_CASE("antiprompt manager - reset/clear") {
    bl::llama::AntipromptManager am;
    am.addAntiprompt("cancel");

    CHECK(am.feedGeneratedText("cance").empty());  // Partial match, so false
    am.reset();  // Reset the manager's antiprompts state
    CHECK(am.feedGeneratedText("cancel") == "cancel"); // Should match, since the state was reset

    am.clear();  // Clear the manager's antiprompts
    CHECK(am.feedGeneratedText("cancel").empty()); // Should match, since the prompts are gone

    am.addAntiprompt("cancel");// add the antiprompt again
    CHECK(am.feedGeneratedText("cancel!") == "cancel!");
}

TEST_CASE("antiprompt manager - reset/clear") {
    bl::llama::AntipromptManager am;
    am.addAntiprompt("\nUser:");
    am.addAntiprompt(" User:");

    am.feedGeneratedText("  Hello! It's nice to meet you. Is there something I can help you with or would you like to chat? ");
    am.feedGeneratedText(" \n\n");
    am.feedGeneratedText("User");
    auto s = am.feedGeneratedText(":");
    CHECK(s == "\nUser:");
}
