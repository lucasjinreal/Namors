use std::sync::RwLock;

use mistralrs::{Loader, VisionLoader, VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig};

use super::vision_model::VisionModelBuilderExt;

pub trait VisionLoaderBuilderExt {
    fn build_custom(self) -> Box<dyn Loader>;
}

impl VisionLoaderBuilderExt for VisionLoaderBuilder {
    fn build_custom(self) -> Box<dyn Loader> {
        let loader: Box<dyn VisionModelBuilderExt> = VisionModelBuilderExt::new();

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
