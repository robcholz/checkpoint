use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::slice;

#[derive(Clone)]
struct TensorInfo {
    ptr: usize,
    numel: usize,
    dtype: String,
}

fn tensor_info(tensor: &Bound<'_, PyAny>, expected_numel: Option<usize>) -> PyResult<TensorInfo> {
    let is_contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if !is_contiguous {
        return Err(PyValueError::new_err("Rust replay requires contiguous tensors"));
    }

    let device = tensor.getattr("device")?;
    let device_type: String = device.getattr("type")?.extract()?;
    if device_type != "cpu" {
        return Err(PyValueError::new_err("Rust replay requires CPU tensors"));
    }

    let numel: usize = tensor.call_method0("numel")?.extract()?;
    if let Some(expected) = expected_numel {
        if numel != expected {
            return Err(PyValueError::new_err(format!(
                "tensor numel mismatch: expected {expected}, got {numel}"
            )));
        }
    }

    let ptr: usize = tensor.call_method0("data_ptr")?.extract()?;
    if ptr == 0 && numel != 0 {
        return Err(PyRuntimeError::new_err("tensor has null data_ptr"));
    }

    let dtype = tensor.getattr("dtype")?.str()?.to_str()?.to_owned();
    Ok(TensorInfo { ptr, numel, dtype })
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

unsafe fn adamw_f32(
    param: *mut f32,
    grad: *const f32,
    exp_avg: *mut f32,
    exp_avg_sq: *mut f32,
    numel: usize,
    step: u64,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    maximize: bool,
) {
    let p = slice::from_raw_parts_mut(param, numel);
    let g = slice::from_raw_parts(grad, numel);
    let m = slice::from_raw_parts_mut(exp_avg, numel);
    let v = slice::from_raw_parts_mut(exp_avg_sq, numel);
    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2_sqrt = (1.0 - beta2.powi(step as i32)).sqrt();
    let step_size = lr / bias_correction1;

    for i in 0..numel {
        let mut grad_i = g[i];
        if maximize {
            grad_i = -grad_i;
        }
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad_i;
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad_i * grad_i;
        if weight_decay != 0.0 {
            p[i] *= 1.0 - lr * weight_decay;
        }
        let denom = v[i].sqrt() / bias_correction2_sqrt + eps;
        p[i] -= step_size * (m[i] / denom);
    }
}

unsafe fn adamw_bf16(
    param: *mut u16,
    grad: *const u16,
    exp_avg: *mut u16,
    exp_avg_sq: *mut u16,
    numel: usize,
    step: u64,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    maximize: bool,
) {
    let p = slice::from_raw_parts_mut(param, numel);
    let g = slice::from_raw_parts(grad, numel);
    let m = slice::from_raw_parts_mut(exp_avg, numel);
    let v = slice::from_raw_parts_mut(exp_avg_sq, numel);
    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2_sqrt = (1.0 - beta2.powi(step as i32)).sqrt();
    let step_size = lr / bias_correction1;

    for i in 0..numel {
        let mut grad_i = bf16_to_f32(g[i]);
        if maximize {
            grad_i = -grad_i;
        }
        let mut p_i = bf16_to_f32(p[i]);
        let mut m_i = bf16_to_f32(m[i]);
        let mut v_i = bf16_to_f32(v[i]);
        m_i = beta1 * m_i + (1.0 - beta1) * grad_i;
        v_i = beta2 * v_i + (1.0 - beta2) * grad_i * grad_i;
        if weight_decay != 0.0 {
            p_i *= 1.0 - lr * weight_decay;
        }
        let denom = v_i.sqrt() / bias_correction2_sqrt + eps;
        p_i -= step_size * (m_i / denom);
        p[i] = f32_to_bf16(p_i);
        m[i] = f32_to_bf16(m_i);
        v[i] = f32_to_bf16(v_i);
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adamw_update(
    py: Python<'_>,
    param: Bound<'_, PyAny>,
    grad: Bound<'_, PyAny>,
    exp_avg: Bound<'_, PyAny>,
    exp_avg_sq: Bound<'_, PyAny>,
    step: u64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    maximize: bool,
) -> PyResult<u64> {
    let param_info = tensor_info(&param, None)?;
    let grad_info = tensor_info(&grad, Some(param_info.numel))?;
    let exp_avg_info = tensor_info(&exp_avg, Some(param_info.numel))?;
    let exp_avg_sq_info = tensor_info(&exp_avg_sq, Some(param_info.numel))?;

    if param_info.dtype != grad_info.dtype
        || param_info.dtype != exp_avg_info.dtype
        || param_info.dtype != exp_avg_sq_info.dtype
    {
        return Err(PyTypeError::new_err(format!(
            "all tensors must share dtype; got param={}, grad={}, exp_avg={}, exp_avg_sq={}",
            param_info.dtype, grad_info.dtype, exp_avg_info.dtype, exp_avg_sq_info.dtype
        )));
    }

    let next_step = step + 1;
    let dtype = param_info.dtype.clone();
    let lr = lr as f32;
    let beta1 = beta1 as f32;
    let beta2 = beta2 as f32;
    let eps = eps as f32;
    let weight_decay = weight_decay as f32;
    let numel = param_info.numel;

    let loop_dtype = dtype.clone();
    py.allow_threads(move || unsafe {
        match loop_dtype.as_str() {
            "torch.float32" => adamw_f32(
                param_info.ptr as *mut f32,
                grad_info.ptr as *const f32,
                exp_avg_info.ptr as *mut f32,
                exp_avg_sq_info.ptr as *mut f32,
                numel,
                next_step,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                maximize,
            ),
            "torch.bfloat16" => adamw_bf16(
                param_info.ptr as *mut u16,
                grad_info.ptr as *const u16,
                exp_avg_info.ptr as *mut u16,
                exp_avg_sq_info.ptr as *mut u16,
                numel,
                next_step,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                maximize,
            ),
            _ => {}
        }
    });

    match dtype.as_str() {
        "torch.float32" | "torch.bfloat16" => Ok(next_step),
        other => Err(PyTypeError::new_err(format!(
            "unsupported dtype for Rust replay: {other}"
        ))),
    }
}

#[pymodule]
fn _gockpt_rust_replay(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adamw_update, m)?)?;
    Ok(())
}
