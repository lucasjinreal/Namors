# Namors

Welcome to Namors.

Namors is a native Rust portable MLLM inference framework, our ultimate goal is to make small VLMs runable anywhere, you can built your toolchains or applications with Namors by calling it's inside model without any further painfulness.

Our goal is to make Namors focus on:

- Candle & mistral.rs as the engine core (computation, tensor etc);
- Optimizations on Small models, written in pure fast Rust;
- VLMs and Omni model is our goal, helpfully we can using Rust deploy MLLMs simply;
- Quantization, we would explore many ways to run int8 or int4 for fastest inference.

The first goal would be inference Namo-R1 in pure Rust: https://github.com/lucasjinreal/Namo-R1/

**models support (or going to support):**

- [X] Qwen2.5 VL;
- [ ] Namo-R1-v1 (50%);
- [ ] Spark-TTS;

If you want adding new LLM based model portable inference, create an issue and start & fork our repo!

## Updates

- `2025-03-11`: 🔥🔥 Runing Qwen2.5-VL initially supported in Namors based on mistral.rs! Namo model are under going;
- `2025-03-01`: The repo created.

## QuickStart

To use `Namors` to run any portable MLLMs, you can simply run:

```
cargo build --release --features metal
```

Be note, if you on macOS, using metal, if you have GPU, you can use cuda.

Then run:

```
./target/release/Namors
```

You can run the model directly. The main code suppose you have model under `checkpoints/Qwen2.5-VL-3B-Instruct`.

You can set your model downloaded path at `main.rs`

## Inference Engine

We choosing mistral.rs as inference engine, because of several reasons:

- we need Rust rather than C++, Cpp is horiable, so we do not use llama.cpp;
- candle is the core, but candle is not really provides a fast / rapid development for feature request / new model / documentation;
- mistral.rs provides speed gurantee and relatively rich documentation, besides, it provides many example model that looks easy to adopt.

We will keep adding more models, and contribution to mistral.rs to make this framework more suitable for portable MLLM inference engine!
