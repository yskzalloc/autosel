# AUTOSEL - Linux Kernel Backport Classifier

AUTOSEL is a tool designed to automatically classify whether Linux kernel commits should be backported to stable trees. It leverages machine learning and large language models to analyze commits and provide recommendations.

## Features

- Automated analysis of kernel commits to determine backport suitability
- Support for multiple LLM providers (Claude, OpenAI, NVIDIA, Kiro)
- Embeddings-based similar commit retrieval
- Vector storage for efficient commit analysis
- Batch processing capabilities
- Detailed explanations for backport decisions

## Requirements

- Rust (2024 edition)
- Linux kernel repository (local)
- API keys for supported LLM providers

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/sashal/autosel.git
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

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
