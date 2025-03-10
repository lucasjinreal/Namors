use mistralrs::{Loader, VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig};
use tokio::sync::RwLock;

use crate::vision_model::CustomVisionModelBuilder;

pub struct CustomVisionLoaderBuilder(VisionLoaderBuilder);

impl CustomVisionLoaderBuilder {
    pub fn new(
        config: VisionSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
    ) -> Self {
        Self(VisionLoaderBuilder::new(
            config,
            chat_template,
            tokenizer_json,
            model_id,
        ))
    }

    pub fn build(self, loader: VisionLoaderType) -> Box<dyn Loader> {
        let loader: Box<dyn CustomVisionModelLoader> = match loader {
            VisionLoaderType::Phi4MM => Box::new(Phi4MMLoader),
        };
        Box::new(VisionLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            xlora_model_id: None,
            xlora_order: None,
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
            from_uqff: RwLock::new(None),
        })
    }
}
