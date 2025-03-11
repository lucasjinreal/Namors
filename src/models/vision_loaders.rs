use std::sync::Arc;

use anyhow::Result;
use mistralrs::{
    layers::Conv3dConfig,
    paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata},
    pipeline::{
        isq::IsqModelLoader, loaders::NonMappedSubModel, DeviceMappedModelLoader,
        NormalLoadingMetadata, Processor, VisionModel, VisionModelLoader,
    },
    vision_models::{preprocessor_config::PreProcessorConfig, processor_config::ProcessorConfig},
    AutoDeviceMapParams, DType, VisionPromptPrefixer,
};
use mistralrs_quant::ShardedVarBuilder;
use regex::Regex;
use std::fmt::Debug;

use crate::models::qwen2_5vl::inputs_processor::Qwen2VLProcessor;
use crate::models::qwen2_5vl::Config as Qwen2VLConfig;
use crate::models::qwen2_5vl::Qwen2VLModel;

macro_rules! bias_if {
    ($cond:expr, $size:expr) => {
        if $cond {
            $size
        } else {
            0
        }
    };
}

// ======================== Qwen25VL Loader

pub struct Qwen25VLLoader;

pub struct Qwen25VLPrefixer;

impl VisionPromptPrefixer for Qwen25VLPrefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        format!(
            "{}{}{}{prompt}",
            Qwen2VLProcessor::VISION_START,
            Qwen2VLProcessor::IMAGE_PAD,
            Qwen2VLProcessor::VISION_END
        )
    }
}

impl VisionModelLoader for Qwen25VLLoader {
    fn load(
        &self,
        config: &str,
        _use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let config: Qwen2VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(Qwen2VLModel::new(
            &config,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, _use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let config: Qwen2VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen2VLProcessor::new(max_edge))
    }
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        let config: Qwen2VLConfig = serde_json::from_str(config)?;
        // We only apply device mapping to text model
        Ok(config.num_hidden_layers)
    }
    fn supports_paged_attention(&self) -> bool {
        false
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Qwen25VLPrefixer)
    }
}

impl IsqModelLoader for Qwen25VLLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Qwen25VLLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &cfg.vision_config;
            let grid_t = max_num_images / cfg.temporal_patch_size;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };
        let img_seq_len = img_seq_len * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len;
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &cfg.vision_config;
            let grid_t = max_num_images / cfg.temporal_patch_size;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };

        let max_vision_attn = {
            let cfg = &cfg.vision_config;
            (max_batch_size * max_num_images) * cfg.num_heads * img_seq_len * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;
        let text_elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let patch_merger = {
            let cfg = &cfg.vision_config;
            let hidden_size = cfg.embed_dim * cfg.spatial_merge_size.pow(2);

            let mlp0 = hidden_size * hidden_size + hidden_size;
            let mlp2 = hidden_size * cfg.hidden_size + cfg.hidden_size;

            let ln_q = cfg.embed_dim + bias_if!(true, cfg.embed_dim);

            mlp0 + mlp2 + ln_q
        };

        let patch_embed = {
            let cfg = &cfg.vision_config;
            let conv_cfg = Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let kernel_sizes = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size];
            cfg.in_channels * cfg.embed_dim / conv_cfg.groups
                * kernel_sizes[0]
                * kernel_sizes[1]
                * kernel_sizes[2]
        };

        let encoder_layer = {
            let cfg = &cfg.vision_config;
            let norm1 = cfg.embed_dim + bias_if!(true, cfg.embed_dim);
            let norm2 = cfg.embed_dim + bias_if!(true, cfg.embed_dim);

            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let mlp_hidden_dim = (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize;
            let fc1 = cfg.embed_dim * mlp_hidden_dim + mlp_hidden_dim;
            let fc2 = cfg.embed_dim * mlp_hidden_dim + cfg.embed_dim;

            let qkv = cfg.embed_dim * cfg.embed_dim * 3 + cfg.embed_dim * 3;
            let out = cfg.embed_dim * cfg.embed_dim + cfg.embed_dim;

            norm1 + norm2 + fc1 + fc2 + qkv + out
        };

        let elems =
            text_elems + patch_merger + patch_embed + encoder_layer * cfg.vision_config.depth;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}
