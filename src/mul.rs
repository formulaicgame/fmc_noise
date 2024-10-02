use std::simd::{LaneCount, Simd, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoiseNode, NoiseNodeSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn mul_1d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Mul {
        left_source,
        right_source,
    } = &node.settings
    else {
        unreachable!()
    };

    unsafe {
        return (left_source.function_1d)(&left_source, x)
            * (right_source.function_1d)(&right_source, x);
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn mul_2d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>, y: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Mul {
        left_source,
        right_source,
    } = &node.settings
    else {
        unreachable!()
    };

    unsafe {
        return (left_source.function_2d)(&left_source, x, y)
            * (right_source.function_2d)(&right_source, x, y);
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn mul_3d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Mul {
        left_source,
        right_source,
    } = &node.settings
    else {
        unreachable!()
    };

    unsafe {
        return (left_source.function_3d)(&left_source, x, y, z)
            * (right_source.function_3d)(&right_source, x, y, z);
    }
}
