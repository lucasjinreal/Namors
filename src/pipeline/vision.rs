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

    // 自定义 build 方法
    pub fn build(self, loader: VisionLoaderType) -> Box<dyn Loader> {
        // 可在这里添加预处理逻辑
        println!("Custom build process starting...");
        
        // 调用原有 build 方法
        let mut original = self.0.build(loader);
        
        // 可在这里添加后处理逻辑
        // 例如：original.set_some_property(...)
        
        original
    }
}