use std::simd::num::SimdFloat;

use multiversion::multiversion;

use crate::NoisePipeline;

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn abs<const N: usize>(pipeline: &mut NoisePipeline<N>) {
    let noise = pipeline.results.pop().unwrap();
    pipeline.results.push(noise.abs());
    pipeline.next();
}
