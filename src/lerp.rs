use std::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};

use multiversion::multiversion;

use crate::NoisePipeline;

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn lerp<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let low = pipeline.results.pop().unwrap();
    let high = pipeline.results.pop().unwrap();
    let selector = pipeline.results.pop().unwrap();

    // This is just a special proprety of the -1..1 range. It's shifted up to be 0..1
    let interpolation = selector.mul_add(Simd::splat(0.5), Simd::splat(1.0));
    let result = (high - low).mul_add(interpolation, low);

    pipeline.results.push(result);
    pipeline.next();
}
