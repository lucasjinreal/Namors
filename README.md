# Namors

Welcome to Namors. 

Namors is a native Rust portable MLLM inference framework, our ultimate goal is to make small VLMs runable anywhere, you can built your toolchains or applications with Namors by calling it's inside model without any further painfulness.

Our goal is to make Namors focus on:

- Candle as the engine core (computation, tensor etc);
- Optimizations on Small models, written in pure fast Rust;
- VLMs and Omni model is our goal, helpfully we can using Rust deploy MLLMs simply;
- Quantization, we would explore many ways to run int8 or int4 for fastest inference.

The first goal would be inference Namo-R1 in pure Rust: https://github.com/lucasjinreal/Namo-R1/

## Status

Namors still in early development.

