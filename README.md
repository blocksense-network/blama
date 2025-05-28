# Blama - Blocksense Llama

A wrapper around [llama.cpp](https://github.com/ggml-org/llama.cpp) that provides a server
with verifiable inference capabilities. Blama enables verifiable AI inference,
ensuring transparency and trust in model outputs.

## Features

- **High Performance**: Built on top of the optimized llama.cpp engine
- **RESTful API**: Easy-to-use HTTP server interface
- **Model Support**: Compatible with GGUF format models

## Quick Start

### Prerequisites

- C++ compiler with C++17 support
- CMake 3.14+
- Git

### Usage

1. **Start the server:**
```bash
./blama-server path/to/your/model.gguf
```

2. **Make complete text requests:**
```bash
curl -X POST http://localhost:7331/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": 'The first man to',
    "max_tokens": 100
  }'
```

3. **Verify completion results:**
```bash
curl -X POST http://localhost:7331/verify_completion \
  -H "Content-Type: application/json" \
  -d '{
    "request": <Here should be added the request to /complete>,
    "response": <Here should be added the response from /complete>
  }'
```

## API Reference

Read more in the document [here](docs/design/Server-API.md).

## Verification System

Blama implements a verification system that ensures the model predictions are correct
based on the output logits of each token generation.

### How It Works

1. Each inference request generates an array of token step generation results.
Each token step has an array of logits (top 10) taken from the context.

2. The same request + response then is send back for verification

3. Each verification request will create the same model
and fill the context with the response's token steps.
During the context filling we'll produce again the same
token steps but with the logits from the current context.

4. Compare the the logits from the request and those returned during context filling.
The algorithm can be checked [here](inference/code/llama/LogitComparer.cpp)


## Supported Models

- Any GGUF-compatible model that is compatible with llama.cpp

## Development

### Building from Source

```bash
# List available presets
cmake --list-presets

# Configure with a preset
cmake --preset debug

# Build with a preset
cmake --build --preset debug
```

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the high-performance inference engine
- Meta AI for the Llama model architecture
- The open source community for contributions and feedback

## Support

- **Issues**: [GitHub Issues](https://github.com/blocksense-network/blama/issues)

---

**Note**: This project is under active development. APIs may change between versions.
