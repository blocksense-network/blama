# Blama webservice API

> **_NOTE:_** All properties with leading "*" mean they are required.
>
> For example - ***model**

## Complete text - /completions

### Request

- ***model** - Id of the model to use.
- ***prompt** - The prompt to gnerate completions for.
- **max_completion_tokens** - The maximum number of tokens that can be generated if invalid token is not generated.
- **seed** - The seed provided to the sampler. If not provided will select random seed.
- **suffix** - The suffix that comes after the completion of the prompt.
- **temperature** - What sampling temperature to use
- **top_p** - An alternative to sampling with temperature. The model will the tokens which have ***top_p*** probability mass. It's not recommended to be used with ***temperture***

```json
{
  "model": "llama3.2-8b",
  "prompt": "Once upon a time in a small village,",
  "max_completion_tokens": 50,
  "suffix": " And that's how the story ends.",
  "temperature": 0.7,
}
```

### Response

- **model** - The model used for completion
- **created** - The Unix timestamp of completion's creation
- **output** - The completion's response
  - **content** - The content of the completion
  - **finish_reason** - Can be either "stop"(if natural stop point was hit), "length"(if the max tokens count from request was hit)
  - ***tokens_data** - List of all generated tokens and their corresponding top 10 logits at the time. Each element has
    - **token** - tokenId
    - **logits** - Vector with *tokenId* and it's *logit value*
- **seed** - The seed used for completion's sampling
- **object** - The object type, always "text_completion"

```json
{
  "model": "llama3.2-8b",
  "created": 1716123456,
  "text": " there lived a curious young boy who loved to explore the forest. Every day, he would venture deeper, discovering hidden paths and ancient trees.",
  "seed": 12345,
  "object": "text_completion"
}
```

### How to achieve with Blama

```c++
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>


...
auto requestBody = parseAndValidate(body); // this function doesn't exist ATM
bl::llama::Model model(requestBody.model, {});
bl::llama::Instance instance(model, {});
instance.warmup();
auto& session = instance.startSession({
  .rngSeed = requestBody.seed,
  .temperature = requestBody.temperature,
  .topP = requestBody.topP
  });
session.setInitialPrompt(model.vocab().tokenize(requestBody.prompt, true, true));
auto tokens_data = session.complete({
  .suffix = requestBody.suffix,
  .maxTokens = requestBody.maxCompletionTokens
})

std::string content;
for (auto& p : tokens_data) {
  content += model.vocab().tokenToString(p.token);
}
```

## Chat complete - /chat/completions

### Request

- ***model** - Id of the model to use.
- ***messages** - A list of the conversation messages. Each message should have:
  - **role** - Message's owner role
  - **content** - Message's textual content
- **max_completion_tokens** - The maximum number of tokens that can be generated if invalid token is not generated.
- **seed** - The seed provided to the sampler. If not provided will select random seed.
- **temperature** - What sampling temperature to use
- **top_p** - An alternative to sampling with temperature. The model will the tokens which have ***top_p*** probability mass.

```json
{
  "model": "llama3.2-8b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Can you tell me a fun fact about space?"
    }
  ],
  "max_completion_tokens": 100,
  "seed": 42
}
```

### Response

- **model** - The model used for completion
- **created** - The Unix timestamp of completion's creation
- **formatted_chat** - Full chat after applying chat format
- **output** - The completion's response
  - **content** - The content of the completion
  - **role** - The role of content's owner
  - **finish_reason** - Can be either "stop"(if natural stop point was hit), "length"(if the max tokens count from request was hit)
  - ***tokens_data** - List of all generated tokens and their corresponding top 10 logits at the time. Each element has
    - **token** - tokenId
    - **logits** - Vector with *tokenId* and it's *logit value*
- **seed** - The seed used for completion's sampling
- **object** - The object type, always "chat.completion"

```json
{
  "model": "llama3.2-8b",
  "created": 1716132562,
  "output": {
    "content": "Sure! Did you know that one day on Venus is longer than its entire year? Venus rotates so slowly that it takes about 243 Earth days to complete one rotation, while it only takes 225 Earth days to orbit the Sun.",
    "role": "assistant",
    "finish_reason": "stop"
  },
  "seed": 42,
  "object": "chat.completion"
}
```

### How to achieve with Blama

```c++
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>
#include <llama/ChatFormat.hpp>


...
auto requestBody = parseAndValidate(body); // this function doesn't exist ATM
bl::llama::Model model(requestBody.model, {});
bl::llama::Instance instance(model, {});
instance.warmup();

bl::llama::ChatFormat::Params chatParams = bl::llama::ChatFormat::getChatParams(model);
chatParams.roleAssistant = "Assistant"; // Check we actually need this
bl::llama::ChatFormat chatFormat(chatParams);
auto formattedChat = chatFormat.formatChat(requestBody.messages, true);

auto& session = instance.startSession({
  .rngSeed = requestBody.seed,
  .temperature = requestBody.temperature,
  .topP = requestBody.topP
});
session.setInitialPrompt(model.vocab().tokenize(formattedChat, true, true));
auto tokens_data = session.complete({
  .maxTokens = requestBody.maxCompletionTokens
})

std::string content;
for (auto& p : tokens_data) {
  content += model.vocab().tokenToString(p.token);
}
```

## Embeddings - /embeddings

### Request

- ***model** - Id of the model to use.
- ***input** - The input text to embed

```json
{
  "model": "e5-mistral-7b-instruct",
  "input": "Blocksense is making verifiable AI inference API."
}
```

### Response

- **embedding** - The embedding vector which is a list of floats. Length depends on the model

```json
{
  "embedding": [-0.012345, 0.098765, -0.023456, 0.045678, ..., 0.007891]
}
```

### How to achieve with Blama

```c++
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/InstanceEmbedding.hpp>

...
auto requestBody = parseAndValidate(body); // this function doesn't exist ATM
bl::llama::Model model(requestBody.model, {});
// Params as rngSeed, temperature, topP have to be added
// Since the instance create a sampler we can create them here.
// Otherwise, we can move their initialization to the Session.
bl::llama::InstanceEmbedding instance(model, {});
auto embedding = instance.getEmbeddingVector(requestBody.input);
```

## Verification - /verify

### Request

- ***model** - Id of the model to use.
- ***prompt** - The inital prompt that was generated for
- ***tokens_data** - List of all generated tokens and their corresponding top 10 logits at the time. Each element has
  - **token** - tokenId
  - **logits** - Vector with *tokenId* and it's *logit value*

```json
{
  "model": "llama3.2-8b",
  ""
  "tokens_data": [
    {
      "token": 123,
      "logits": [
        { "token": 123, "logit": 12.34 },
        ...
        { "token": 456, "logit": 10.56 }
      ]
    },
    ...
    {
      "token": 456,
      "logits": [
        { "token": 456, "logit": 11.11 },
        ...
        { "token": 321, "logit": 10.01 }
      ]
    }
  ]
}
```

### Response

Here we have 2 options:

Directly return the similarity value, so the Reporter will run the similarity check based on the metrics:

- **similarity** - A floating point value between 0-1 that determines how close it is to the input **tokens_data**

```json
{
  "similarity": 0.9823
}
```

Or run the check in the sequencer, so we'll need to return the metrics:

- **tokens_data** - List of the same tokens as the generated ones, but with the new logits which are produced after we feed the context with the input tokens

```json
{
  "tokens_data": [
    {
      "token": 123,
      "logits": [
        { "token": 123, "logit": 12.34 },
        ...
        { "token": 456, "logit": 10.56 }
      ]
    },
    ...
    {
      "token": 456,
      "logits": [
        { "token": 456, "logit": 11.11 },
        ...
        { "token": 321, "logit": 10.01 }
      ]
    }
  ]
}
```


### How to achieve with Blama

```c++
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/LogitComparer.hpp>

...
auto requestBody = parseAndValidate(body); // this function doesn't exist ATM

bl::llama::Model model(modelGguf, {}, modelLoadProgressCallback);
bl::llama::Instance instance(model, {});
auto& session = instance.startSession({});
session.setInitialPrompt(model.vocab().tokenize(requestBody.prompt, true, true));;
auto respTokensData = sessionCpu.fillCtx(requestBody.tokens_data);

// If we want to run the metrics on the reporter we should run also:
bl::llama::MetricsAggregator metricsAgg;
float similarity = 0;
for (size_t i = 0; i < iRes.size(); i++) {
    auto m = bl::llama::LogitComparer::compare(requestBody.tokens_data[i].logits, respTokensData[i].logits);
    similarity = metricsAgg.pushAndVerify({ &m, 1 });
}

```
