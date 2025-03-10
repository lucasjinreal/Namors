pub struct CustomVisionModelBuilder(VisionModelBuilder);

impl CustomVisionModelBuilder {
    pub fn new(model_id: impl ToString, loader_type: VisionLoaderType) -> Self {
        Self {
            model_id: model_id.to_string(),
            use_flash_attn: cfg!(feature = "flash-attn"),
            topology: None,
            write_uqff: None,
            from_uqff: None,
            prompt_chunksize: None,
            chat_template: None,
            tokenizer_json: None,
            max_edge: None,
            loader_type,
            dtype: ModelDType::Auto,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            isq: None,
            max_num_seqs: 32,
            with_logging: false,
            device_mapping: None,
            calibration_file: None,
            imatrix: None,
        }
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
