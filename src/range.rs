use std::simd::{prelude::*, LaneCount, Simd, StdFloat, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoiseNode, NoiseNodeSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn range_1d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Range {
        high,
        low,
        selector,
        low_source,
        high_source,
    } = &node.settings
    else {
        unreachable!()
    };

    let high = Simd::splat(*high);
    let low = Simd::splat(*low);

    unsafe {
        let selection_noise = (selector.function_1d)(&selector, x);
        let low_noise = (low_source.function_1d)(&low_source, x);
        let high_noise = (high_source.function_1d)(&high_source, x);

        let high_clipped = selection_noise.simd_gt(high);
        let low_clipped = selection_noise.simd_lt(low);

        let mut interpolation = (selection_noise - low) / (high - low);
        interpolation = (high_noise - low_noise).mul_add(interpolation, low_noise);

        let mut result = high_clipped.select(high_noise, interpolation);
        result = low_clipped.select(low_noise, result);
        return result;
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn range_2d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Range {
        high,
        low,
        selector,
        low_source,
        high_source,
    } = &node.settings
    else {
        unreachable!()
    };

    let high = Simd::splat(*high);
    let low = Simd::splat(*low);

    unsafe {
        let selection_noise = (selector.function_2d)(&selector, x, y);
        let low_noise = (low_source.function_2d)(&low_source, x, y);
        let high_noise = (high_source.function_2d)(&high_source, x, y);

        let high_clipped = selection_noise.simd_gt(high);
        let low_clipped = selection_noise.simd_lt(low);

        let mut interpolation = (selection_noise - low) / (high - low);
        interpolation = (high_noise - low_noise).mul_add(interpolation, low_noise);

        let mut result = high_clipped.select(high_noise, interpolation);
        result = low_clipped.select(low_noise, result);
        return result;
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn range_3d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Range {
        high,
        low,
        selector,
        low_source,
        high_source,
    } = &node.settings
    else {
        unreachable!()
    };

    let high = Simd::splat(*high);
    let low = Simd::splat(*low);

    unsafe {
        let selection_noise = (selector.function_3d)(&selector, x, y, z);
        let low_noise = (low_source.function_3d)(&low_source, x, y, z);
        let high_noise = (high_source.function_3d)(&high_source, x, y, z);

        let high_clipped = selection_noise.simd_gt(high);
        let low_clipped = selection_noise.simd_lt(low);

        let mut interpolation = (selection_noise - low) / (high - low);
        interpolation = (high_noise - low_noise).mul_add(interpolation, low_noise);

        let mut result = high_clipped.select(high_noise, interpolation);
        result = low_clipped.select(low_noise, result);
        return result;
    }
}
