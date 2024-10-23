use std::simd::{prelude::*, LaneCount, Simd, StdFloat, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoisePipeline, NoiseSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn range<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let low_noise = pipeline.results.pop().unwrap();
    let high_noise = pipeline.results.pop().unwrap();
    let selector_noise = pipeline.results.pop().unwrap();

    let node = pipeline.current_node();

    let NoiseSettings::Range { low, high } = node.settings else {
        unreachable!()
    };

    let low = Simd::splat(low);
    let high = Simd::splat(high);

    let low_clipped = selector_noise.simd_lt(low);
    let high_clipped = selector_noise.simd_gt(high);

    let mut interpolation = (selector_noise - low) / (high - low);
    interpolation = (high_noise - low_noise).mul_add(interpolation, low_noise);

    let mut result = high_clipped.select(high_noise, interpolation);
    result = low_clipped.select(low_noise, result);

    pipeline.results.push(result);
    pipeline.next();
}
