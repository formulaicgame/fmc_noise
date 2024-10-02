use std::simd::{LaneCount, Simd, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoiseNode, NoiseNodeSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn square_1d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Square { source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        let source_result = (source.function_1d)(&source, x);
        return source_result * source_result;
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn square_2d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Square { source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        let source_result = (source.function_2d)(&source, x, y);
        return source_result * source_result;
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn square_3d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Square { source } = &node.settings else {
        unreachable!()
    };

    unsafe {
        let source_result = (source.function_3d)(&source, x, y, z);
        return source_result * source_result;
    }
}
