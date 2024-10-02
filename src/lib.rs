#![feature(portable_simd)]
#![allow(private_bounds)]

use std::simd::prelude::*;

use multiversion::{multiversion, target::selected_target};
use std::simd::{LaneCount, SupportedLaneCount};

mod abs;
mod add;
mod clamp;
mod constant;
mod fbm;
mod gradient;
mod lerp;
mod min_and_max;
mod mul;
mod perlin;
mod range;
mod simplex;
mod square;

// TODO: Make a cargo feature "f64", makes it compile to f64 instead of f32
//if cfg(f64)
//type Float = f64;
//type Int = i64;
//else
//type Float = f32;
//type Int = i32;

// TODO: Find some way to make this Copy? You often use the same noise many places and clone makes
// it noisy.
#[derive(Clone, Debug)]
pub struct Noise {
    settings: Box<NoiseSettings>,
}

impl Noise {
    pub fn simplex(frequency: impl Into<Frequency>, seed: i32) -> Self {
        return Self {
            settings: Box::new(NoiseSettings::Simplex {
                seed,
                frequency: frequency.into(),
            }),
        };
    }

    pub fn perlin(frequency: impl Into<Frequency>, seed: i32) -> Self {
        return Self {
            settings: Box::new(NoiseSettings::Perlin {
                seed,
                frequency: frequency.into(),
            }),
        };
    }

    pub fn constant(value: f32) -> Self {
        return Self {
            settings: Box::new(NoiseSettings::Constant { value }),
        };
    }

    /// Fractal Brownian Motion (layered noise)
    pub fn fbm(mut self, octaves: u32, gain: f32, lacunarity: f32) -> Self {
        // The amplitude gets pre scaled so that it can skip normalizing the result.
        // e.g. if the gain is 0.5 and there are 2 octaves, the amplitude would be 1 + 0.5 = 1.5
        // after both octaves are combined normally. Instead, we set the initial amplitude to be
        // 1/1.5 == 2/3 and so the second octave's amplitude becomes 2/3 * 0.5 = 1/3 and we end up
        // with a normalized result naturally.
        let mut amp = gain;
        let mut scale = 1.0;
        for _ in 1..octaves {
            scale += amp;
            amp *= gain;
        }
        scale = 1.0 / scale;

        self.settings = Box::new(NoiseSettings::Fbm {
            octaves,
            gain,
            lacunarity,
            scale,
            source: self.settings,
        });
        self
    }

    /// Convert the noise to absolute values.
    pub fn abs(mut self) -> Self {
        self.settings = Box::new(NoiseSettings::Abs {
            source: self.settings,
        });
        self
    }

    /// Add two noises together, the result is not normalized.
    pub fn add(mut self, other: Self) -> Self {
        self.settings = Box::new(NoiseSettings::AddNoise {
            left: self.settings,
            right: other.settings,
        });
        self
    }

    // TODO: Remove and replace with noise::constant, same for mul_value
    /// Add a value to the noise
    pub fn add_value(mut self, value: f32) -> Self {
        self.settings = Box::new(NoiseSettings::AddValue {
            value,
            source: self.settings,
        });
        self
    }

    /// Clamp the noise values between min and max
    pub fn clamp(mut self, min: f32, max: f32) -> Self {
        self.settings = Box::new(NoiseSettings::Clamp {
            min,
            max,
            source: self.settings,
        });
        self
    }

    /// Take the max value between the two noises
    pub fn max(mut self, other: Self) -> Self {
        self.settings = Box::new(NoiseSettings::Max {
            left: self.settings,
            right: other.settings,
        });
        self
    }

    /// Take the min value between the two noises
    pub fn min(mut self, other: Self) -> Self {
        self.settings = Box::new(NoiseSettings::Min {
            left: self.settings,
            right: other.settings,
        });
        self
    }

    // TODO: Convert to just 'mul' and take a noise
    /// Multiply the noise by a value.
    pub fn mul_value(mut self, value: f32) -> Self {
        self.settings = Box::new(NoiseSettings::MulValue {
            value,
            source: self.settings,
        });
        self
    }

    pub fn lerp(mut self, high: Self, low: Self) -> Self {
        self.settings = Box::new(NoiseSettings::Lerp {
            selector_source: self.settings,
            high_source: high.settings,
            low_source: low.settings,
        });
        self
    }

    pub fn range(mut self, high: f32, low: f32, high_noise: Self, low_noise: Self) -> Self {
        self.settings = Box::new(NoiseSettings::Range {
            high,
            low,
            selector_source: self.settings,
            high_source: high_noise.settings,
            low_source: low_noise.settings,
        });
        self
    }

    pub fn square(mut self) -> Self {
        self.settings = Box::new(NoiseSettings::Square {
            source: self.settings,
        });
        self
    }

    pub fn generate_1d(&self, x: f32, width: usize) -> (Vec<f32>, f32, f32) {
        generate_1d(self, x, width)
    }

    pub fn generate_2d(&self, x: f32, y: f32, width: usize, height: usize) -> (Vec<f32>, f32, f32) {
        generate_2d(self, x, y, width, height)
    }

    pub fn generate_3d(
        &self,
        x: f32,
        y: f32,
        z: f32,
        width: usize,
        height: usize,
        depth: usize,
    ) -> (Vec<f32>, f32, f32) {
        generate_3d(self, x, y, z, width, height, depth)
    }
}

#[derive(Clone, Copy, Debug)]
struct Frequency {
    x: f32,
    y: f32,
    z: f32,
}

impl From<[f32; 3]> for Frequency {
    fn from(value: [f32; 3]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<[f32; 2]> for Frequency {
    fn from(value: [f32; 2]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[1],
        }
    }
}

impl From<f32> for Frequency {
    fn from(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }
}

#[derive(Clone, Debug)]
enum NoiseSettings {
    Simplex {
        seed: i32,
        frequency: Frequency,
    },
    Perlin {
        seed: i32,
        frequency: Frequency,
    },
    Constant {
        value: f32,
    },
    Fbm {
        /// Total number of octaves
        /// The number of octaves control the amount of detail in the noise function.
        /// Adding more octaves increases the detail, with the drawback of increasing the calculation time.
        octaves: u32,
        /// Gain is a multiplier on the amplitude of each successive octave.
        /// i.e. A gain of 2.0 will cause each octave to be twice as impactful on the result as the
        /// previous one.
        gain: f32,
        /// Lacunarity is multiplied by the frequency for each successive octave.
        /// i.e. a value of 2.0 will cause each octave to have double the frequency of the previous one.
        lacunarity: f32,
        // Automatically derived scaling factor.
        scale: f32,
        source: Box<NoiseSettings>,
    },
    Abs {
        source: Box<NoiseSettings>,
    },
    AddNoise {
        left: Box<NoiseSettings>,
        right: Box<NoiseSettings>,
    },
    AddValue {
        value: f32,
        source: Box<NoiseSettings>,
    },
    Clamp {
        min: f32,
        max: f32,
        source: Box<NoiseSettings>,
    },
    Lerp {
        selector_source: Box<NoiseSettings>,
        high_source: Box<NoiseSettings>,
        low_source: Box<NoiseSettings>,
    },
    Max {
        left: Box<NoiseSettings>,
        right: Box<NoiseSettings>,
    },
    Min {
        left: Box<NoiseSettings>,
        right: Box<NoiseSettings>,
    },
    MulValue {
        value: f32,
        source: Box<NoiseSettings>,
    },
    Range {
        high: f32,
        low: f32,
        selector_source: Box<NoiseSettings>,
        high_source: Box<NoiseSettings>,
        low_source: Box<NoiseSettings>,
    },
    Square {
        source: Box<NoiseSettings>,
    },
}

#[derive(Debug)]
pub(crate) enum NoiseNodeSettings<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    Simplex {
        seed: i32,
        frequency: Frequency,
    },
    Perlin {
        seed: i32,
        frequency: Frequency,
    },
    Constant {
        value: f32,
    },
    Fbm {
        octaves: u32,
        gain: f32,
        lacunarity: f32,
        scale: f32,
        source: Box<NoiseNode<N>>,
    },
    Abs {
        source: Box<NoiseNode<N>>,
    },
    AddNoise {
        left_source: Box<NoiseNode<N>>,
        right_source: Box<NoiseNode<N>>,
    },
    AddValue {
        value: f32,
        source: Box<NoiseNode<N>>,
    },
    Clamp {
        min: f32,
        max: f32,
        source: Box<NoiseNode<N>>,
    },
    MaxNoise {
        left_source: Box<NoiseNode<N>>,
        right_source: Box<NoiseNode<N>>,
    },
    MinNoise {
        left_source: Box<NoiseNode<N>>,
        right_source: Box<NoiseNode<N>>,
    },
    MulValue {
        value: f32,
        source: Box<NoiseNode<N>>,
    },
    Lerp {
        selector: Box<NoiseNode<N>>,
        low_source: Box<NoiseNode<N>>,
        high_source: Box<NoiseNode<N>>,
    },
    Range {
        low: f32,
        high: f32,
        selector: Box<NoiseNode<N>>,
        low_source: Box<NoiseNode<N>>,
        high_source: Box<NoiseNode<N>>,
    },
    Square {
        source: Box<NoiseNode<N>>,
    },
}

#[derive(Debug)]
pub(crate) struct NoiseNode<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub settings: NoiseNodeSettings<N>,
    pub function_1d: unsafe fn(node: &NoiseNode<N>, x: Simd<f32, N>) -> Simd<f32, N>,
    pub function_2d:
        unsafe fn(node: &NoiseNode<N>, x: Simd<f32, N>, y: Simd<f32, N>) -> Simd<f32, N>,
    pub function_3d: unsafe fn(
        node: &NoiseNode<N>,
        x: Simd<f32, N>,
        y: Simd<f32, N>,
        z: Simd<f32, N>,
    ) -> Simd<f32, N>,
}

impl<const N: usize> From<&Box<NoiseSettings>> for NoiseNode<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn from(value: &Box<NoiseSettings>) -> Self {
        match value.as_ref() {
            NoiseSettings::Simplex { seed, frequency } => Self {
                settings: NoiseNodeSettings::Simplex {
                    seed: *seed,
                    frequency: *frequency,
                },
                function_1d: crate::simplex::simplex_1d(),
                function_2d: crate::simplex::simplex_2d(),
                function_3d: crate::simplex::simplex_3d(),
            },
            NoiseSettings::Perlin { seed, frequency } => Self {
                settings: NoiseNodeSettings::Perlin {
                    seed: *seed,
                    frequency: *frequency,
                },
                function_1d: crate::simplex::simplex_1d(),
                function_2d: crate::perlin::perlin_2d(),
                function_3d: crate::perlin::perlin_3d(),
            },
            NoiseSettings::Constant { value } => Self {
                settings: NoiseNodeSettings::Constant { value: *value },
                function_1d: crate::constant::constant_1d(),
                function_2d: crate::constant::constant_2d(),
                function_3d: crate::constant::constant_3d(),
            },
            NoiseSettings::Fbm {
                octaves,
                gain,
                lacunarity,
                scale,
                source,
            } => Self {
                settings: NoiseNodeSettings::Fbm {
                    octaves: *octaves,
                    gain: *gain,
                    lacunarity: *lacunarity,
                    scale: *scale,
                    source: Box::new(Self::from(source)),
                },
                function_1d: crate::fbm::fbm_1d(),
                function_2d: crate::fbm::fbm_2d(),
                function_3d: crate::fbm::fbm_3d(),
            },
            NoiseSettings::Abs { source } => Self {
                settings: NoiseNodeSettings::Abs {
                    source: Box::new(Self::from(source)),
                },
                function_1d: crate::abs::abs_1d(),
                function_2d: crate::abs::abs_2d(),
                function_3d: crate::abs::abs_3d(),
            },
            NoiseSettings::AddNoise { left, right } => Self {
                settings: NoiseNodeSettings::AddNoise {
                    left_source: Box::new(Self::from(left)),
                    right_source: Box::new(Self::from(right)),
                },
                function_1d: crate::add::add_1d(),
                function_2d: crate::add::add_2d(),
                function_3d: crate::add::add_3d(),
            },
            NoiseSettings::AddValue { value, source } => Self {
                settings: NoiseNodeSettings::AddValue {
                    value: *value,
                    source: Box::new(Self::from(source)),
                },
                function_1d: crate::add::add_value_1d(),
                function_2d: crate::add::add_value_2d(),
                function_3d: crate::add::add_value_3d(),
            },
            NoiseSettings::Clamp { min, max, source } => Self {
                settings: NoiseNodeSettings::Clamp {
                    min: *min,
                    max: *max,
                    source: Box::new(Self::from(source)),
                },
                function_1d: crate::clamp::clamp_1d(),
                function_2d: crate::clamp::clamp_2d(),
                function_3d: crate::clamp::clamp_3d(),
            },
            NoiseSettings::Max { left, right } => Self {
                settings: NoiseNodeSettings::MaxNoise {
                    left_source: Box::new(Self::from(left)),
                    right_source: Box::new(Self::from(right)),
                },
                function_1d: crate::min_and_max::max_1d(),
                function_2d: crate::min_and_max::max_2d(),
                function_3d: crate::min_and_max::max_3d(),
            },
            NoiseSettings::Min { left, right } => Self {
                settings: NoiseNodeSettings::MinNoise {
                    left_source: Box::new(Self::from(left)),
                    right_source: Box::new(Self::from(right)),
                },
                function_1d: crate::min_and_max::min_1d(),
                function_2d: crate::min_and_max::min_2d(),
                function_3d: crate::min_and_max::min_3d(),
            },
            NoiseSettings::MulValue { value, source } => Self {
                settings: NoiseNodeSettings::MulValue {
                    value: *value,
                    source: Box::new(Self::from(source)),
                },
                function_1d: crate::mul::mul_value_1d(),
                function_2d: crate::mul::mul_value_2d(),
                function_3d: crate::mul::mul_value_3d(),
            },
            NoiseSettings::Lerp {
                selector_source,
                low_source,
                high_source,
            } => NoiseNode {
                settings: NoiseNodeSettings::Lerp {
                    selector: Box::new(Self::from(selector_source)),
                    low_source: Box::new(Self::from(low_source)),
                    high_source: Box::new(Self::from(high_source)),
                },
                function_1d: crate::lerp::lerp_1d(),
                function_2d: crate::lerp::lerp_2d(),
                function_3d: crate::lerp::lerp_3d(),
            },
            NoiseSettings::Range {
                high,
                low,
                selector_source,
                high_source,
                low_source,
            } => NoiseNode {
                settings: NoiseNodeSettings::Range {
                    low: *low,
                    high: *high,
                    selector: Box::new(Self::from(selector_source)),
                    low_source: Box::new(Self::from(low_source)),
                    high_source: Box::new(Self::from(high_source)),
                },
                function_1d: crate::range::range_1d(),
                function_2d: crate::range::range_2d(),
                function_3d: crate::range::range_3d(),
            },
            NoiseSettings::Square { source } => Self {
                settings: NoiseNodeSettings::Square {
                    source: Box::new(Self::from(source)),
                },
                function_1d: crate::square::square_1d(),
                function_2d: crate::square::square_2d(),
                function_3d: crate::square::square_3d(),
            },
        }
    }
}

#[multiversion(targets = "simd")]
fn generate_1d(noise: &Noise, x: f32, width: usize) -> (Vec<f32>, f32, f32) {
    const N: usize = if let Some(size) = selected_target!().suggested_simd_width::<f32>() {
        size
    } else {
        1
    };

    let noise_node = NoiseNode::<N>::from(&noise.settings);

    let start_x = x;

    let mut min_s = Simd::splat(f32::MAX);
    let mut max_s = Simd::splat(f32::MIN);
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    let mut result = Vec::with_capacity(width);
    unsafe {
        result.set_len(width);
    }
    let vector_width = N;
    let remainder = width % vector_width;
    let mut x_arr = Vec::with_capacity(vector_width);
    unsafe {
        x_arr.set_len(vector_width);
    }
    for i in (0..vector_width).rev() {
        x_arr[i] = start_x + i as f32;
    }

    let mut i = 0;
    let mut x = Simd::from_slice(&x_arr);
    for _ in 0..width / vector_width {
        let f = unsafe { (noise_node.function_1d)(&noise_node, x) };
        max_s = max_s.simd_max(f);
        min_s = min_s.simd_min(f);
        f.copy_to_slice(&mut result[i..]);
        i += vector_width;
        x += Simd::splat(vector_width as f32);
    }
    if remainder != 0 {
        let f = unsafe { (noise_node.function_1d)(&noise_node, x) };
        for j in 0..remainder {
            let n = f[j];
            unsafe {
                *result.get_unchecked_mut(i) = n;
            }
            if n < min {
                min = n;
            }
            if n > max {
                max = n;
            }
            i += 1;
        }
    }
    for i in 0..vector_width {
        if min_s[i] < min {
            min = min_s[i];
        }
        if max_s[i] > max {
            max = max_s[i];
        }
    }
    (result, min, max)
}

#[multiversion(targets = "simd")]
fn generate_2d(noise: &Noise, x: f32, y: f32, width: usize, height: usize) -> (Vec<f32>, f32, f32) {
    const N: usize = if let Some(size) = selected_target!().suggested_simd_width::<f32>() {
        size
    } else {
        1
    };

    let noise_node = NoiseNode::<N>::from(&noise.settings);
    let start_x = y;
    let start_y = x;

    let mut min_s = Simd::splat(f32::MAX);
    let mut max_s = Simd::splat(f32::MIN);
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    let mut result = Vec::with_capacity(width * height);
    unsafe {
        result.set_len(width * height);
    }
    let mut y = Simd::splat(start_y);
    let mut i = 0;
    let vector_width = N;
    let remainder = width % vector_width;
    let mut x_arr = Vec::with_capacity(vector_width);
    unsafe {
        x_arr.set_len(vector_width);
    }
    for i in (0..vector_width).rev() {
        x_arr[i] = start_x + i as f32;
    }
    for _ in 0..height {
        let mut x = Simd::from_slice(&x_arr);
        for _ in 0..width / vector_width {
            let f = unsafe { (noise_node.function_2d)(&noise_node, x, y) };
            max_s = max_s.simd_max(f);
            min_s = min_s.simd_min(f);
            f.copy_to_slice(&mut result[i..]);
            i += vector_width;
            x += Simd::splat(vector_width as f32);
        }
        if remainder != 0 {
            let f = unsafe { (noise_node.function_2d)(&noise_node, x, y) };
            for j in 0..remainder {
                let n = f[j];
                unsafe {
                    *result.get_unchecked_mut(i) = n;
                }
                if n < min {
                    min = n;
                }
                if n > max {
                    max = n;
                }
                i += 1;
            }
        }
        y += Simd::splat(1.0);
    }
    for i in 0..vector_width {
        if min_s[i] < min {
            min = min_s[i];
        }
        if max_s[i] > max {
            max = max_s[i];
        }
    }
    (result, min, max)
}

#[multiversion(targets = "simd")]
fn generate_3d(
    noise: &Noise,
    x: f32,
    y: f32,
    z: f32,
    width: usize,
    height: usize,
    depth: usize,
) -> (Vec<f32>, f32, f32) {
    const N: usize = if let Some(size) = selected_target!().suggested_simd_width::<f32>() {
        size
    } else {
        1
    };
    let noise_node = NoiseNode::<N>::from(&noise.settings);

    let start_x = x;
    let start_y = y;
    let start_z = z;

    let mut min_s = Simd::splat(f32::MAX);
    let mut max_s = Simd::splat(f32::MIN);
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    let mut result = Vec::with_capacity(width * height * depth);
    unsafe {
        result.set_len(width * height * depth);
    }
    let mut i = 0;
    let vector_width = N;
    let remainder = height % vector_width;
    let mut y_arr = Vec::with_capacity(vector_width);
    unsafe {
        y_arr.set_len(vector_width);
    }
    for i in (0..vector_width).rev() {
        y_arr[i] = start_y + i as f32;
    }

    // TODO: This loop in loop system is maybe not good? Try a flat design where "overflowing"
    // values of the first axis is transfered to the second, and same for second to third every
    // iteration.
    let mut x = Simd::splat(start_x);
    for _ in 0..width {
        let mut z = Simd::splat(start_z);
        for _ in 0..depth {
            let mut y = Simd::from_slice(&y_arr);
            for _ in 0..height / vector_width {
                let f = unsafe { (noise_node.function_3d)(&noise_node, x, y, z) };
                max_s = max_s.simd_max(f);
                min_s = min_s.simd_min(f);
                f.copy_to_slice(&mut result[i..]);
                i += vector_width;
                y = y + Simd::splat(vector_width as f32);
            }
            if remainder != 0 {
                let f = unsafe { (noise_node.function_3d)(&noise_node, x, y, z) };
                for j in 0..remainder {
                    let n = f[j];
                    unsafe {
                        *result.get_unchecked_mut(i) = n;
                    }
                    if n < min {
                        min = n;
                    }
                    if n > max {
                        max = n;
                    }
                    i += 1;
                }
            }
            z = z + Simd::splat(1.0);
        }
        x = x + Simd::splat(1.0);
    }
    for i in 0..vector_width {
        if min_s[i] < min {
            min = min_s[i];
        }
        if max_s[i] > max {
            max = max_s[i];
        }
    }
    (result, min, max)
}
