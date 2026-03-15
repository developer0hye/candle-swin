//! Detailed error analysis for BiRefNet backbone comparison.
//! Run with: cargo test --test error_analysis -- --nocapture

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

fn load_test_case(name: &str) -> HashMap<String, Tensor> {
    let path = format!("test-data/{name}.safetensors");
    let data = std::fs::read(&path).unwrap_or_else(|_| panic!("missing test data: {path}"));
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();
    let device = &Device::Cpu;
    tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F64 => DType::F64,
                safetensors::Dtype::I64 => DType::I64,
                safetensors::Dtype::U32 => DType::U32,
                dt => panic!("unsupported dtype {dt:?} for tensor {name}"),
            };
            let tensor = Tensor::from_raw_buffer(view.data(), dtype, view.shape(), device).unwrap();
            (name.to_string(), tensor)
        })
        .collect()
}

fn vb_from_params(data: &HashMap<String, Tensor>) -> (VarMap, VarBuilder<'_>) {
    let varmap = VarMap::new();
    {
        let mut data_map = varmap.data().lock().unwrap();
        for (key, tensor) in data {
            if let Some(param_name) = key.strip_prefix("param.") {
                data_map.insert(param_name.to_string(), Var::from_tensor(tensor).unwrap());
            }
        }
    }
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    (varmap, vb)
}

#[test]
fn analyze_birefnet_backbone_error() -> Result<()> {
    let data = load_test_case("birefnet_swin_backbone");
    let (_varmap, vb) = vb_from_params(&data);

    let config = candle_swin::swin_transformer::SwinTransformerConfig::large();
    let model = candle_swin::SwinTransformer::new(&config, vb)?;
    let outputs = model.forward(&data["input"])?;

    println!("\n{}", "=".repeat(70));
    println!("BiRefNet Swin-L Backbone: Candle vs PyTorch Error Analysis");
    println!("{}\n", "=".repeat(70));

    for i in 0..4 {
        let key = format!("output_{i}");
        let expected = &data[&key];
        let actual = &outputs[i];

        let diff = actual
            .to_dtype(DType::F64)?
            .sub(&expected.to_dtype(DType::F64)?)?
            .abs()?;

        let max_diff: f64 = diff.flatten_all()?.max(0)?.to_scalar::<f64>()?;
        let mean_diff: f64 = diff.mean_all()?.to_scalar::<f64>()?;

        let abs_expected = expected.to_dtype(DType::F64)?.abs()?;
        let max_val: f64 = abs_expected.flatten_all()?.max(0)?.to_scalar::<f64>()?;
        let mean_val: f64 = abs_expected.mean_all()?.to_scalar::<f64>()?;

        let relative_max: f64 = max_diff / max_val;
        let relative_mean: f64 = mean_diff / mean_val;

        println!("Stage {i}: {:?}", actual.dims());
        println!("  Absolute error:  max={max_diff:.6e}  mean={mean_diff:.6e}");
        println!("  Output range:    max_abs={max_val:.4}  mean_abs={mean_val:.4}");
        println!("  Relative error:  max={relative_max:.6e}  mean={relative_mean:.6e}");
        println!();
    }

    Ok(())
}
