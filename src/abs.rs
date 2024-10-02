use std::simd::{num::SimdFloat, LaneCount, SupportedLaneCount};

use multiversion::multiversion;

use crate::NoisePipeline;

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn abs<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let noise = pipeline.results.pop().unwrap();
    pipeline.results.push(noise.abs());
    pipeline.next();
}
