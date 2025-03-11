use mistralrs::{
    best_device, initialize_logging, AutoDeviceMapParams, DefaultSchedulerMethod, DeviceMapSetting, Loader, MistralRsBuilder, Model, SchedulerConfig, VisionLoaderType, VisionModelBuilder, VisionSpecificConfig
};

use super::vision::VisionLoaderBuilderExt;

pub trait VisionModelBuilderExt {
    fn build_custom(self) -> anyhow::Result<Model>;
}

impl VisionModelBuilderExt for VisionModelBuilder {
    fn build_custom(self) -> anyhow::Result<Model> {
        let config = VisionSpecificConfig {
            use_flash_attn: self.use_flash_attn,
            prompt_chunksize: self.prompt_chunksize,
            topology: self.topology,
            write_uqff: self.write_uqff,
            from_uqff: self.from_uqff,
            max_edge: self.max_edge,
            calibration_file: self.calibration_file,
            imatrix: self.imatrix,
        };

        if self.with_logging {
            initialize_logging();
        }

        let loader = VisionLoaderBuilderExt::new(
            config,
            self.chat_template,
            self.tokenizer_json,
            Some(self.model_id),
        )
        .build(self.loader_type);

        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            self.hf_revision,
            self.token_source,
            &self.dtype,
            &best_device(self.force_cpu)?,
            !self.with_logging,
            self.device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_vision())),
            self.isq,
            None,
        )?;

        let scheduler_method = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(self.max_num_seqs.try_into()?),
        };

        let runner = MistralRsBuilder::new(pipeline, scheduler_method)
            .with_no_kv_cache(false)
            .with_gemm_full_precision_f16(true)
            .with_no_prefix_cache(false);

        Ok(Model::new(runner.build()))
    }
}
