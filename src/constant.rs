use std::simd::{LaneCount, Simd, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoisePipeline, NoiseSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn constant<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let node = pipeline.current_node();

    let NoiseSettings::Constant { value } = node.settings else {
        unreachable!()
    };

    let result = Simd::splat(value);
    pipeline.results.push(result);
    pipeline.next();
}
