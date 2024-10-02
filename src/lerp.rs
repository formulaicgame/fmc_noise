use std::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};

use multiversion::multiversion;

use crate::{NoiseNode, NoiseNodeSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn lerp_1d<const N: usize>(node: &NoiseNode<N>, x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Lerp {
        selector,
        low_source,
        high_source,
    } = &node.settings
    else {
        unreachable!()
    };

    unsafe {
        let high = (high_source.function_1d)(&high_source, x);
        let low = (low_source.function_1d)(&low_source, x);
        // This is just a special proprety of the -1..1 range. It's shifted up to be 0..1
        let interpolation =
            (selector.function_1d)(&selector, x).mul_add(Simd::splat(0.5), Simd::splat(1.0));
        return (high - low).mul_add(interpolation, low);
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn lerp_2d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Lerp {
        selector,
        low_source,
        high_source,
    } = &node.settings
    else {
        unreachable!()
    };

    unsafe {
        let range = (high_source.function_2d)(&high_source, x, y)
            - (low_source.function_2d)(&low_source, x, y);
        let interpolation =
            ((selector.function_2d)(&selector, x, y) + Simd::splat(1.0)) * Simd::splat(0.5);
        return range * interpolation;
    }
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn lerp_3d<const N: usize>(
    node: &NoiseNode<N>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Lerp {
        selector,
        low_source,
        high_source,
    } = &node.settings
    else {
        unreachable!()
    };

    unsafe {
        let low_noise = (low_source.function_3d)(&low_source, x, y, z);
        let high_noise = (high_source.function_3d)(&high_source, x, y, z);

        let interpolation =
            ((selector.function_3d)(&selector, x, y, z) + Simd::splat(1.0)) * Simd::splat(0.5);

        return (high_noise - low_noise).mul_add(interpolation, low_noise);
    }
}
