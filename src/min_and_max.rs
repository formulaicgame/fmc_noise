use std::simd::{prelude::*, LaneCount, SupportedLaneCount};

use multiversion::multiversion;

use crate::NoisePipeline;

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn max<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let left = pipeline.results.pop().unwrap();
    let right = pipeline.results.pop().unwrap();
    let result = left.simd_max(right);
    pipeline.results.push(result);
    pipeline.next();
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn min<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let left = pipeline.results.pop().unwrap();
    let right = pipeline.results.pop().unwrap();
    let result = left.simd_min(right);
    pipeline.results.push(result);
    pipeline.next();
}
