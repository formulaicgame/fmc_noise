use std::simd::{prelude::*, LaneCount, Simd, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoisePipeline, NoiseSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn clamp<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let node = pipeline.current_node();
    let NoiseSettings::Clamp { min, max } = node.settings else {
        unreachable!()
    };

    let noise = pipeline.results.pop().unwrap();
    let result = noise.simd_clamp(Simd::splat(min), Simd::splat(max));
    pipeline.results.push(result);
    pipeline.next();
}
