#![feature(portable_simd)]

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

// TODO: Make a cargo feature "f64", makes it compile with f64 instead of f32
//if cfg(f64)
//type Float = f64;
//type Int = i64;
//else
//type Float = f32;
//type Int = i32;

/// Defines a noise
///
/// # Example
/// ```rust
/// // Ridged fractal noise with values from 0.5 to 1.5
/// let x = 0.0;
/// let y = 0.0;
/// let z = 0.0;
/// let width = 16;
/// let height = 16;
/// let depth = 16;
///
/// let octaves = 5;
/// let gain = 0.5;
/// let lacunarity = 2.0;
///
/// let seed = 42;
///
/// let noise = Noise::perlin(0.01)
///     .seed(seed)
///     .fbm(octaves, gain, lacunarity)
///     .abs()
///     .add(Noise::constant(0.5))
///     .generate_3d(x, y, z, width, height, depth);
/// ```
#[derive(Clone, Debug)]
pub struct Noise {
    seed: u64,
    pipeline: Vec<NoiseSettings>,
}

impl Noise {
    /// [Simplex noise](https://en.wikipedia.org/wiki/Simplex_noise)  
    ///
    /// # Example
    /// ```rust
    /// let noise = Noise::simplex(0.01);
    /// // Is equivalent to
    /// let noise = Noise::simplex(Frequency { x: 0.01, y: 0.01, z: 0.01 });
    /// ```
    pub fn simplex(frequency: impl Into<Frequency>) -> Self {
        return Self {
            seed: 0,
            pipeline: vec![NoiseSettings::Simplex {
                frequency: frequency.into(),
            }],
        };
    }

    /// [Perlin noise](https://en.wikipedia.org/wiki/Perlin_noise)  
    ///
    /// # Example
    /// See [`Frequency`] for convenient [`From`] implementations.
    /// ```rust
    /// let noise = Noise::perlin(0.01);
    /// // Is equivalent to
    /// let noise = Noise::perlin(Frequency { x: 0.01, y: 0.01, z: 0.01 });
    /// ```
    pub fn perlin(frequency: impl Into<Frequency>) -> Self {
        return Self {
            seed: 0,
            pipeline: vec![NoiseSettings::Perlin {
                frequency: frequency.into(),
            }],
        };
    }

    /// A constant number, useful for shifting values.
    ///
    /// # Example
    /// ```rust
    /// // Noise moved from -1..1 to 0..1
    /// let noise = Noise::simplex(0.01)
    ///     .mul(Noise::constant(0.5))
    ///     .add(Noise::constant(0.5));
    /// ```
    pub fn constant(value: f32) -> Self {
        return Self {
            seed: 0,
            pipeline: vec![NoiseSettings::Constant { value }],
        };
    }

    /// Set the seed of the random number generator.
    ///
    /// # Example
    /// ```rust
    /// let noise_left = Noise::simplex(0.01).seed(1);
    /// let noise_right = Noise::simplex(0.01).seed(2);
    /// // In cases like this the seed is always taken from the noise the method is called on,
    /// // so the seed here will be 1.
    /// let noise = noise_left.add(noise_right);
    /// ```
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed as u64;
        self
    }

    /// [Fractal Brownian Motion](https://en.wikipedia.org/wiki/Fractional_Brownian_motion)
    ///
    /// Computes `octaves` layers of noise and adds them together, normalizing the result. Each
    /// consecutive octave has its frequency multiplied by `lacunarity` and its amplitude
    /// multiplied by `gain`.
    ///
    /// # Example
    /// ```rust
    /// // Common values for gain and lacunarity. Each octave has half the amplitude and twice
    /// // the frequency, addding finer detail to the noise.
    /// let noise = Noise::simplex(0.01).fbm(5, 0.5, 2.0);
    /// ```
    pub fn fbm(mut self, octaves: u32, gain: f32, lacunarity: f32) -> Self {
        assert!(octaves > 0, "There must be 1 or more octaves");

        // The amplitude gets pre-scaled so that we can skip normalizing the result.
        // e.g. if the gain is 0.5 and there are 2 octaves, the amplitude would be 1 + 0.5 = 1.5
        // when both octaves are combined normally. Instead, we set the initial amplitude to be
        // 1/1.5 == 2/3, the second octave's amplitude becomes 2/3 * 0.5 = 1/3 and we end up with a
        // normalized result naturally.
        let mut amp = gain;
        let mut scaled_amplitude = 1.0;
        for _ in 1..octaves {
            scaled_amplitude += amp;
            amp *= gain;
        }
        scaled_amplitude = 1.0 / scaled_amplitude;

        // It's not possible to apply the lacunarity when executing as we don't know that the
        // noise is included in an fbm node before we reach it in the pipeline. Instead we have to
        // pre-apply the lacunarity to the frequency. The fbm node adds the results of the noises from
        // last to first, so lacunarity is also applied in that order. .i.e. the first noise is the most
        // lacunarized.
        let len = self.pipeline.len();
        for i in (1..octaves).rev() {
            let current_len = self.pipeline.len();
            let settings = &mut self.pipeline[current_len - len];
            match settings {
                NoiseSettings::Simplex { frequency, .. } => {
                    let lacunarity = lacunarity.powi(i as i32 - 1);
                    frequency.x *= lacunarity;
                    frequency.y *= lacunarity;
                    frequency.z *= lacunarity;
                }
                NoiseSettings::Perlin { frequency, .. } => {
                    let lacunarity = lacunarity.powi(i as i32 - 1);
                    frequency.x *= lacunarity;
                    frequency.y *= lacunarity;
                    frequency.z *= lacunarity;
                }
                _ => (),
            }
            self.pipeline.extend_from_within(0..len);
        }

        self.pipeline.push(NoiseSettings::Fbm {
            octaves,
            gain,
            scaled_amplitude,
        });
        self
    }

    /// Computes the absolute value of the noise
    pub fn abs(mut self) -> Self {
        self.pipeline.push(NoiseSettings::Abs);
        self
    }

    /// Add two noises, the result is not normalized.
    pub fn add(mut self, mut other: Self) -> Self {
        self.pipeline.append(&mut other.pipeline);
        self.pipeline.push(NoiseSettings::Add);
        self
    }

    /// Multiply two noises, the result is not normalized.
    pub fn mul(mut self, mut other: Self) -> Self {
        self.pipeline.append(&mut other.pipeline);
        self.pipeline.push(NoiseSettings::Mul);
        self
    }

    /// Clamp the noise between min and max
    pub fn clamp(mut self, min: f32, max: f32) -> Self {
        self.pipeline.push(NoiseSettings::Clamp { min, max });
        self
    }

    /// Take the maximum of the two noises
    pub fn max(mut self, mut other: Self) -> Self {
        self.pipeline.append(&mut other.pipeline);
        self.pipeline.push(NoiseSettings::Max);
        self
    }

    /// Take the minimum of the two noises
    pub fn min(mut self, mut other: Self) -> Self {
        self.pipeline.append(&mut other.pipeline);
        self.pipeline.push(NoiseSettings::Min);
        self
    }

    /// Linearly interpolate between high and low using the input noise.
    /// <div class="warning">The 'self' noise is required to be in the -1..1 range.</div>
    pub fn lerp(mut self, mut low: Self, mut high: Self) -> Self {
        // XXX: Append order is important for result order
        self.pipeline.append(&mut high.pipeline);
        self.pipeline.append(&mut low.pipeline);
        self.pipeline.push(NoiseSettings::Lerp);
        self
    }

    /// Interpolate between the high and low noise. When the input noise is above 'high' it's
    /// clamped to the high noise, and below 'low' to the low noise. When in-between, use the input
    /// noise to linearly interpolate between them.
    /// <div class="warning">The 'self' noise is required to be in the -1..1 range.</div>
    pub fn range(mut self, low: f32, high: f32, mut low_noise: Self, mut high_noise: Self) -> Self {
        // XXX: Append order is important for result order
        self.pipeline.append(&mut high_noise.pipeline);
        self.pipeline.append(&mut low_noise.pipeline);
        self.pipeline.push(NoiseSettings::Range { low, high });
        self
    }

    /// Square the noise, noise²
    pub fn square(mut self) -> Self {
        self.pipeline.push(NoiseSettings::Square);
        self
    }

    /// Generates a line of noise. It also returns the min and max value generated.
    ///
    /// # Example
    /// ```
    /// let width = 16;
    /// let noise = Noise::perlin(0.01).generate_1d(0.0, width);
    /// for x in 0..width {
    ///     let value = noise[x];
    /// }
    /// ```
    pub fn generate_1d(&self, x: f32, width: usize) -> (Vec<f32>, f32, f32) {
        generate_1d(self, x, width)
    }

    /// Generates a plane of noise. It also returns the min and max value generated.
    ///
    /// # Example
    /// ```
    /// let width = 16;
    /// let height = 16;
    /// let noise = Noise::perlin(0.01).generate_2d(0.0, 0.0, width, height);
    /// for x in 0..width {
    ///     for y in 0..height {
    ///         // This is how you should index the generated values
    ///         let index = x * height + y;
    ///         let value = noise[index];
    ///         // If width == height and it's a power of 2 you can also index like this
    ///         // 16 is 2^4 so we use 4 to shift. At 32 it would be 5, and 64, 6 etc...
    ///         let index = x << 4 | y;
    ///         let value = noise[index];
    ///     }
    /// }
    /// ```
    pub fn generate_2d(&self, x: f32, y: f32, width: usize, height: usize) -> (Vec<f32>, f32, f32) {
        generate_2d(self, x, y, width, height)
    }

    /// Generates a cube of noise. It also returns the min and max value generated.
    ///
    /// # Example
    /// ```
    /// let width = 16;
    /// let height = 16;
    /// let depth = 16;
    /// let noise = Noise::perlin(0.01).generate_3d(0.0, 0.0, 0.0, width, height, depth);
    /// for x in 0..width {
    ///     for z in 0..depth {
    ///         for y in 0..height {
    ///             // This is how you should index the generated values
    ///             let index = x * depth * height + z * height + y;
    ///             let value = noise[index];
    ///             // If width == height == depth and it's a power of 2 you can also index like this
    ///             // 16 is 2^4 so we use 4/8 to shift. At 32 it would be 5/10, and 64, 6/12 etc...
    ///             let index = x << 8 | z << 4 | y;
    ///             let value = noise[index];
    ///         }
    ///     }
    /// }
    /// ```
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

/// The frequencies of a noise.
///
/// # Example
/// ```
/// // A convenient way to set the frequencies, as you usually want them to be the same.
/// // If you intend to generate 1d/2d noise it doesn't matter if the remaining axis are
/// // set.
/// let noise = Noise::perlin(0.01);
/// // Is equivalent to
/// let noise = Noise::perlin(Frequency {
///     x: 0.01,
///     y: 0.01,
///     z: 0.01
/// });
/// // Or
/// let noise = Noise::perlin(Frequency::new_3d(0.01, 0.01, 0.01));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Frequency {
    /// First dimension
    pub x: f32,
    /// Third dimension
    pub y: f32,
    /// Second dimension
    pub z: f32,
}

impl Frequency {
    pub fn new_1d(x: f32) -> Self {
        Self { x, y: 0.0, z: 0.0 }
    }

    pub fn new_2d(x: f32, z: f32) -> Self {
        Self { x, y: 0.0, z }
    }

    pub fn new_3d(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
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
        frequency: Frequency,
    },
    Perlin {
        frequency: Frequency,
    },
    Constant {
        value: f32,
    },
    Fbm {
        // Total number of octaves
        // The number of octaves control the amount of detail in the noise function.
        // Adding more octaves increases the detail, with the drawback of increasing the calculation time.
        octaves: u32,
        // Gain is a multiplier on the amplitude of each successive octave.
        // i.e. A gain of 2.0 will cause each octave to be twice as impactful on the result as the
        // previous one.
        gain: f32,
        // Automatically derived amplitude scaling factor.
        scaled_amplitude: f32,
    },
    Abs,
    Add,
    Mul,
    Clamp {
        min: f32,
        max: f32,
    },
    Max,
    Min,
    Lerp,
    Range {
        low: f32,
        high: f32,
    },
    Square,
}

#[derive(Debug)]
struct NoisePipeline<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    rng: Rng,
    index: usize,
    pipeline: Vec<NoiseNode<N>>,
    results: Vec<Simd<f32, N>>,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
}

impl<const N: usize> NoisePipeline<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    #[track_caller]
    fn current_node(&self) -> &NoiseNode<N> {
        &self.pipeline[self.index]
    }

    #[inline(always)]
    fn next(&mut self) {
        self.index += 1;
        if self.index == self.pipeline.len() {
            return;
        }
        unsafe { (&self.pipeline[self.index].function)(self) };
    }

    #[inline(always)]
    fn execute(&mut self) -> Simd<f32, N> {
        self.index = 0;
        self.rng.reset();

        unsafe { (self.pipeline[0].function)(self) };
        return self.results.pop().unwrap();
    }

    fn build(noise: &Noise, dimensions: Dimensions) -> Self {
        let mut pipeline = Vec::with_capacity(noise.pipeline.len());

        for settings in noise.pipeline.iter().cloned() {
            let function = match settings {
                NoiseSettings::Simplex { .. } => match dimensions {
                    Dimensions::X => crate::simplex::simplex_1d(),
                    Dimensions::XY => crate::simplex::simplex_2d(),
                    Dimensions::XYZ => crate::simplex::simplex_3d(),
                },
                NoiseSettings::Perlin { .. } => match dimensions {
                    Dimensions::X => crate::simplex::simplex_1d(),
                    Dimensions::XY => crate::perlin::perlin_2d(),
                    Dimensions::XYZ => crate::perlin::perlin_3d(),
                },
                NoiseSettings::Constant { .. } => crate::constant::constant(),
                NoiseSettings::Fbm { .. } => crate::fbm::fbm(),
                NoiseSettings::Abs { .. } => crate::abs::abs(),
                NoiseSettings::Add { .. } => crate::add::add(),
                NoiseSettings::Mul { .. } => crate::mul::mul(),
                NoiseSettings::Clamp { .. } => crate::clamp::clamp(),
                NoiseSettings::Max { .. } => crate::min_and_max::max(),
                NoiseSettings::Min { .. } => crate::min_and_max::min(),
                NoiseSettings::Lerp { .. } => crate::lerp::lerp(),
                NoiseSettings::Range { .. } => crate::range::range(),
                NoiseSettings::Square { .. } => crate::square::square(),
            };
            let noise_node = NoiseNode { settings, function };

            pipeline.push(noise_node)
        }

        NoisePipeline {
            rng: Rng::new(noise.seed),
            index: 0,
            pipeline,
            results: Vec::new(),
            x: Simd::splat(0.0),
            y: Simd::splat(0.0),
            z: Simd::splat(0.0),
        }
    }
}

enum Dimensions {
    X,
    XY,
    XYZ,
}

#[derive(Debug)]
struct NoiseNode<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub settings: NoiseSettings,
    pub function: unsafe fn(pipeline: &mut NoisePipeline<N>),
}

#[multiversion(targets = "simd")]
fn generate_1d(noise: &Noise, x: f32, width: usize) -> (Vec<f32>, f32, f32) {
    const N: usize = if let Some(size) = selected_target!().suggested_simd_width::<f32>() {
        size
    } else {
        1
    };

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
        x_arr[i] = x + i as f32;
    }

    let mut pipeline = NoisePipeline::<N>::build(noise, Dimensions::X);
    pipeline.x = Simd::from_slice(&x_arr);

    let mut i = 0;
    for _ in 0..width / vector_width {
        let f = pipeline.execute();
        max_s = max_s.simd_max(f);
        min_s = min_s.simd_min(f);
        f.copy_to_slice(&mut result[i..]);
        i += vector_width;
        pipeline.x += Simd::splat(vector_width as f32);
    }
    if remainder != 0 {
        let f = pipeline.execute();
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

    let mut min_s = Simd::splat(f32::MAX);
    let mut max_s = Simd::splat(f32::MIN);
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    let mut result = Vec::with_capacity(width * height);
    unsafe {
        result.set_len(width * height);
    }

    let vector_width = N;
    let remainder = width % vector_width;
    let mut x_arr = Vec::with_capacity(vector_width);
    unsafe {
        x_arr.set_len(vector_width);
    }
    for i in (0..vector_width).rev() {
        x_arr[i] = x + i as f32;
    }

    let mut pipeline = NoisePipeline::<N>::build(noise, Dimensions::XY);
    pipeline.y = Simd::splat(y);

    let mut i = 0;
    for _ in 0..height {
        pipeline.x = Simd::from_slice(&x_arr);
        for _ in 0..width / vector_width {
            let f = pipeline.execute();
            max_s = max_s.simd_max(f);
            min_s = min_s.simd_min(f);
            f.copy_to_slice(&mut result[i..]);
            i += vector_width;
            pipeline.x += Simd::splat(vector_width as f32);
        }
        if remainder != 0 {
            let f = pipeline.execute();
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
        pipeline.y += Simd::splat(1.0);
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
        y_arr[i] = y + i as f32;
    }

    let mut pipeline = NoisePipeline::<N>::build(noise, Dimensions::XYZ);

    // TODO: This loop in loop system is maybe not good? Try a flat design where "overflowing"
    // values of the first axis is transfered to the second, and same for second to third every
    // iteration.
    pipeline.x = Simd::splat(x);
    for _ in 0..width {
        pipeline.z = Simd::splat(z);
        for _ in 0..depth {
            pipeline.y = Simd::from_slice(&y_arr);
            for _ in 0..height / vector_width {
                let f = pipeline.execute();
                max_s = max_s.simd_max(f);
                min_s = min_s.simd_min(f);
                f.copy_to_slice(&mut result[i..]);
                i += vector_width;
                pipeline.y += Simd::splat(vector_width as f32);
            }
            if remainder != 0 {
                let f = pipeline.execute();
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
            pipeline.z += Simd::splat(1.0);
        }
        pipeline.x += Simd::splat(1.0);
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

// See WyRand https://github.com/wangyi-fudan/wyhash/blob/master/wyhash.h#L151
#[derive(Debug, Clone)]
struct Rng {
    // We need to be able to reset the seed between each loop of the pipeline so we have to store
    // it.
    seed: u64,
    current_seed: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            current_seed: seed,
        }
    }

    fn next(&mut self) -> i32 {
        let seed = self.current_seed.wrapping_add(0x2d35_8dcc_aa6c_78a5);
        self.current_seed = seed;
        let t = u128::from(seed) * u128::from(seed ^ 0x8bb8_4b93_962e_acc9);
        ((t as u64) ^ (t >> 64) as u64) as i32
    }

    fn reset(&mut self) {
        self.current_seed = self.seed;
    }
}
