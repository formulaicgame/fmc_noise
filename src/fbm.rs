use multiversion::multiversion;

use std::simd::{LaneCount, Simd, SupportedLaneCount};
use crate::{NoiseNode, NoiseNodeSettings};

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn fbm_1d<const N: usize>(node: &NoiseNode<N>, mut x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Fbm {
        octaves,
        gain,
        lacunarity,
        scale,
        source,
    } = &node.settings
    else {
        unreachable!()
    };
    let lacunarity = Simd::splat(*lacunarity);
    let gain = Simd::splat(*gain);
    let mut amplitude = Simd::splat(*scale);
    let mut result = Simd::splat(0.0);

    for _ in 0..(*octaves) {
        let noise = unsafe { (source.function_1d)(&source, x) };
        result += noise * amplitude;
        amplitude *= gain;
        x *= lacunarity;
    }

    result
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn fbm_2d<const N: usize>(
    node: &NoiseNode<N>,
    mut x: Simd<f32, N>,
    mut y: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Fbm {
        octaves,
        gain,
        lacunarity,
        scale,
        source,
    } = &node.settings
    else {
        unreachable!()
    };
    let lacunarity = Simd::splat(*lacunarity);
    let gain = Simd::splat(*gain);
    let mut amplitude = Simd::splat(*scale);
    let mut result = Simd::splat(0.0);

    for _ in 0..(*octaves) {
        let noise = unsafe { (source.function_2d)(&source, x, y) };
        result += noise * amplitude;
        amplitude *= gain;
        x *= lacunarity;
        y *= lacunarity;
    }

    result
}

#[multiversion(targets = "simd", dispatcher = "pointer")]
pub fn fbm_3d<const N: usize>(
    node: &NoiseNode<N>,
    mut x: Simd<f32, N>,
    mut y: Simd<f32, N>,
    mut z: Simd<f32, N>,
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let NoiseNodeSettings::Fbm {
        octaves,
        gain,
        lacunarity,
        scale,
        source,
    } = &node.settings
    else {
        unreachable!()
    };
    let lacunarity = Simd::splat(*lacunarity);
    let gain = Simd::splat(*gain);
    let mut amplitude = Simd::splat(*scale);
    let mut result = Simd::splat(0.0);

    for _ in 0..(*octaves) {
        let noise = unsafe { (source.function_3d)(&source, x, y, z) };
        result += noise * amplitude;
        amplitude *= gain;
        x *= lacunarity;
        y *= lacunarity;
        z *= lacunarity;
    }

    result
}
