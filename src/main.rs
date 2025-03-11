mod models;
mod pipeline;

use std::io::Write;

use anyhow::Result;
use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, Response, TextMessageRole, VisionLoaderType,
    VisionMessages, VisionModelBuilder,
};
use pipeline::vision_model::VisionModelBuilderExt;

const MODEL_ID: &str = "checkpoints/Qwen2.5-VL-3B-Instruct";

#[tokio::main]
async fn main() -> Result<()> {
    // this actually called qwen2.5 vl
    let model = VisionModelBuilder::new(MODEL_ID, VisionLoaderType::Qwen2VL)
        // .with_isq(IsqType::Q4K)
        .with_logging()
        .build_custom()
        .await?;

    let bytes = match reqwest::blocking::get(
        "https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;
    println!("image loaded.");

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "What type of flower is this? Give some fun facts.",
        image,
        &model,
    )?;

    // no-stream mode
    // let response = model.send_chat_request(messages).await?;
    // println!("{}", response.choices[0].message.content.as_ref().unwrap());
    // dbg!(
    //     response.usage.avg_prompt_tok_per_sec,
    //     response.usage.avg_compl_tok_per_sec
    // );

    let mut stream = model.stream_chat_request(messages).await?;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    while let Some(chunk) = stream.next().await {
        if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
            if let Some(ChunkChoice {
                delta:
                    Delta {
                        content: Some(content),
                        ..
                    },
                ..
            }) = choices.first()
            {
                lock.write_all(content.as_bytes())?;
                lock.flush()?;
            };
        } else {
            // Handle errors
        }
    }
    println!("\ndone");

    Ok(())
}
