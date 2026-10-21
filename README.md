# AUTOSEL - Linux Kernel Backport Classifier

AUTOSEL is a tool designed to automatically classify whether Linux kernel commits should be backported to stable trees. It leverages machine learning and large language models to analyze commits and provide recommendations.

## Features

- Automated analysis of kernel commits to determine backport suitability
- Support for multiple LLM providers (Claude, OpenAI, NVIDIA, Kiro, goose)
- Embeddings-based similar commit retrieval
- Vector storage for efficient commit analysis
- Batch processing capabilities
- Detailed explanations for backport decisions

## Requirements

- Rust 1.88 or newer (2024 edition, uses let-chains)
- Linux kernel repository (local)
- API keys for supported LLM providers

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/sashalevin/autosel.git
   cd autosel
   ```

2. Build the project:
   ```
   cargo build --release
   ```

3. Optional: Build with CUDA support for faster embeddings:
   ```
   cargo build --release --features cuda
   ```

## Configuration

Set the following environment variables for the LLM providers you want to use:

- `ANTHROPIC_API_KEY` - For Claude models
- `OPENAI_API_KEY` - For OpenAI models (GPT-4, etc.)
- `NVIDIA_API_KEY` - For NVIDIA models
- `KIRO_API_KEY` - For Kiro (alternatively, use an existing `kiro-cli` browser login)

For Kiro, no API key is needed if you already have an authenticated `kiro-cli` session (browser login). Set `KIRO_API_KEY` for headless/CI environments.

For [goose](https://github.com/block/goose), autosel runs `goose run` in the kernel tree, so goose's own configuration and environment pick the backend (`GOOSE_PROVIDER`, `GOOSE_MODEL`, plus the provider's credentials). A local OpenAI-compatible server such as vLLM works through goose's `openai` provider:

```
GOOSE_PROVIDER=openai GOOSE_MODEL=OpenVINO/Qwen3-8B-int4-ov \
OPENAI_HOST=http://localhost:8000 OPENAI_API_KEY=unused \
./target/release/autosel --kernel-repo ~/path/to/linux --models goose --commit <SHA>
```

goose always offers the model at least one tool, so start vLLM with tool calling enabled (`--enable-auto-tool-choice --tool-call-parser hermes` for Qwen3), or set `GOOSE_TOOLSHIM=true` for a server without it. The autosel prompt alone is over 10k tokens, so the server's `--max-model-len` has to be well above that. By default goose gets no extensions; pass `--goose-builtins developer` to let the agent run git and other shell commands in the kernel tree.

## Usage

Basic usage:
```
./target/release/autosel --kernel-repo ~/path/to/linux --models claude
```

Analyze a specific commit:
```
./target/release/autosel --kernel-repo ~/path/to/linux --models claude --commit <SHA>
```

Analyze multiple commits:
```
./target/release/autosel --kernel-repo ~/path/to/linux --models claude --commits <SHA1> <SHA2> <SHA3>
```

Train the model on existing data:
```
./target/release/autosel --kernel-repo ~/path/to/linux --train
```

Generate and view prompts without sending them:
```
./target/release/autosel --kernel-repo ~/path/to/linux --commit <SHA> --make-prompt ./prompts
```

### Using with Local and Free LLMs

You can use AUTOSEL with local LLMs or free alternatives to commercial APIs using the `--make-prompt` option:

1. Generate prompts without sending to any API:
   ```
   ./target/release/autosel --kernel-repo ~/path/to/linux --commit <SHA> --make-prompt ./my-prompts
   ```

2. This saves structured prompts to the specified directory that you can manually feed to any LLM:
   ```
   cat ./my-prompts/prompt_<SHA>.txt | llm-cli --model my-local-model
   ```

3. Compatible with various local LLM runners:
   - [llama.cpp](https://github.com/ggerganov/llama.cpp)
   - [Ollama](https://ollama.ai/)
   - [LocalAI](https://localai.io/)
   - [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)

4. Workflow example with Ollama:
   ```bash
   # Generate the prompt file for a commit
   ./target/release/autosel --kernel-repo ~/path/to/linux --commit abc123 --make-prompt ./prompts

   # Process with Ollama using a local code-capable model
   cat ./prompts/prompt_abc123.txt | ollama run codellama:7b > result.txt

   # Examine results
   cat result.txt
   ```

5. Tips for using with local LLMs:
   - For best results, use code-specialized models (CodeLlama, WizardCoder, etc.)
   - Larger models (7B+) typically perform better on code analysis tasks
   - Add system prompts to enhance the model's context awareness about kernel development

Batch processing with saved explanations:
```
./target/release/autosel --kernel-repo ~/path/to/linux --models claude,openai --commits <SHA1> <SHA2> --batch ./explanations
```

## How It Works

1. AUTOSEL extracts commit features including message, diff, author information, and files changed
2. It uses embeddings to find similar commits from the historical data
3. The system constructs a prompt for LLMs that includes the commit information and relevant context
4. Multiple LLM providers can analyze the commit in parallel
5. Results are aggregated and a final recommendation is provided

## License

SPDX-License-Identifier: GPL-2.0
(c) 2025, Sasha Levin <sashal@kernel.org>

## Contributing

Contributions are welcome. Open a pull request against `main` at
<https://github.com/sashalevin/autosel>, or file an issue for bugs and feature
requests.

This project follows kernel conventions:

- **Sign your work.** Every commit needs a `Signed-off-by:` line certifying the
  [Developer Certificate of Origin](https://developercertificate.org/).
  `git commit -s` adds it, and CI checks it on pull requests.
- **Commit messages** use imperative subjects, optionally prefixed with the
  subsystem (`classifier: drop redundant kiro default config`). Explain *why*
  in the body, wrapped at 72 columns.
- **One logical change per commit.** Pull requests are merged with rebase, so
  each commit must build and pass tests on its own.
- Run `cargo test` before opening a pull request.

Emailed patches to `sashal@kernel.org` are still accepted, but pull requests
are preferred and get CI coverage.

The classification library lives in a separate repository:
<https://github.com/sashalevin/commit-classifier>.
