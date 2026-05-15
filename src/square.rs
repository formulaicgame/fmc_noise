use multiversion::multiversion;

use crate::NoisePipeline;

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn square<const N: usize>(pipeline: &mut NoisePipeline<N>) {
    let noise = pipeline.results.pop().unwrap();
    pipeline.results.push(noise * noise);
    pipeline.next();
}
