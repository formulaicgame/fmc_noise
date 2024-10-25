use multiversion::multiversion;

use crate::{NoisePipeline, NoiseSettings};
use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn fbm<const N: usize>(pipeline: &mut NoisePipeline<N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    let node = pipeline.current_node();

    let NoiseSettings::Fbm {
        octaves,
        gain,
        first_octave_amplitude,
    } = node.settings
    else {
        unreachable!()
    };

    let gain = Simd::splat(gain);
    let mut amplitude = Simd::splat(first_octave_amplitude);
    let mut result = Simd::splat(0.0);

    for _ in 0..octaves {
        let noise = pipeline.results.pop().unwrap();
        result += noise * amplitude;
        amplitude *= gain;
    }

    pipeline.results.push(result);
    pipeline.next();
}
