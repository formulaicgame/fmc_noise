use std::simd::{LaneCount, SupportedLaneCount};

use multiversion::multiversion;

use crate::NoisePipeline;

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn mul<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let left = pipeline.results.pop().unwrap();
    let right = pipeline.results.pop().unwrap();
    pipeline.results.push(left * right);
    pipeline.next();
}
