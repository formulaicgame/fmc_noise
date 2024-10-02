use std::simd::{prelude::*, LaneCount, Simd, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoiseNode, NoiseNodeSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn clamp_1d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Clamp { min, max, source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_1d)(&source, x).simd_clamp(Simd::splat(*min), Simd::splat(*max));
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn clamp_2d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Clamp { min, max, source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_2d)(&source, x, y)
            .simd_clamp(Simd::splat(*min), Simd::splat(*max));
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn clamp_3d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Clamp { min, max, source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_3d)(&source, x, y, z)
            .simd_clamp(Simd::splat(*min), Simd::splat(*max));
    }
}
