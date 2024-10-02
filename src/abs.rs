use std::simd::{num::SimdFloat, LaneCount, Simd, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoiseNode, NoiseNodeSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn abs_1d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Abs { source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_1d)(&source, x).abs();
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn abs_2d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>, y: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Abs { source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_2d)(&source, x, y).abs();
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn abs_3d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Abs { source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_3d)(&source, x, y, z).abs();
    }
}
