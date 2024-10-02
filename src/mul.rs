use std::simd::{LaneCount, Simd, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoiseNode, NoiseNodeSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn mul_value_1d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::MulValue { value, source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_1d)(&source, x) * Simd::splat(*value);
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn mul_value_2d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::MulValue { value, source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_2d)(&source, x, y) * Simd::splat(*value);
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn mul_value_3d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::MulValue { value, source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        return (source.function_3d)(&source, x, y, z) * Simd::splat(*value);
    }
}
