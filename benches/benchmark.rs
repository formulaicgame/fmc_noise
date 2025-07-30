use criterion::{criterion_group, criterion_main, Criterion};
use fmc_noise::Noise;

const SIZE_2D: usize = 1024;
const SIZE_3D: usize = 101;

fn perlin_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fmc_noise/2d");

    let noise = Noise::perlin(0.01);
    group.bench_function("perlin", |b| {
        b.iter(|| noise.generate_2d(0.0, 0.0, SIZE_2D, SIZE_2D))
    });

    let noise = Noise::perlin(0.01).fbm(2, 0.5, 2.0);
    group.bench_function("perlin fbm 2 octaves", |b| {
        b.iter(|| noise.generate_2d(0.0, 0.0, SIZE_2D, SIZE_2D))
    });

    let noise = Noise::perlin(0.01).fbm(8, 0.5, 2.0);
    group.bench_function("perlin fbm 8 octaves", |b| {
        b.iter(|| noise.generate_2d(0.0, 0.0, SIZE_2D, SIZE_2D))
    });

    group
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(5));
}

fn simplex_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fmc_noise/2d");

    let noise = Noise::simplex(0.01);
    group.bench_function("simplex", |b| {
        b.iter(|| noise.generate_2d(0.0, 0.0, SIZE_2D, SIZE_2D))
    });

    let noise = Noise::simplex(0.01).fbm(2, 0.5, 2.0);
    group.bench_function("simplex fbm 2 octaves", |b| {
        b.iter(|| noise.generate_2d(0.0, 0.0, SIZE_2D, SIZE_2D))
    });

    let noise = Noise::simplex(0.01).fbm(8, 0.5, 2.0);
    group.bench_function("simplex fbm 8 octaves", |b| {
        b.iter(|| noise.generate_2d(0.0, 0.0, SIZE_2D, SIZE_2D))
    });

    group
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(5));
}

fn perlin_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fmc_noise/3d");

    let noise = Noise::perlin(0.01);
    group.bench_function("perlin", |b| {
        b.iter(|| noise.generate_3d(0.0, 0.0, 0.0, SIZE_3D, SIZE_3D, SIZE_3D))
    });

    let noise = Noise::perlin(0.01).fbm(2, 0.5, 2.0);
    group.bench_function("perlin fbm 2 octaves", |b| {
        b.iter(|| noise.generate_3d(0.0, 0.0, 0.0, SIZE_3D, SIZE_3D, SIZE_3D))
    });

    let noise = Noise::perlin(0.01).fbm(8, 0.5, 2.0);
    group.bench_function("perlin fbm 8 octaves", |b| {
        b.iter(|| noise.generate_3d(0.0, 0.0, 0.0, SIZE_3D, SIZE_3D, SIZE_3D))
    });

    group
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(5));
}

fn simplex_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fmc_noise/3d");

    let noise = Noise::simplex(0.01);
    group.bench_function("simplex", |b| {
        b.iter(|| noise.generate_3d(0.0, 0.0, 0.0, SIZE_3D, SIZE_3D, SIZE_3D))
    });

    let noise = Noise::simplex(0.01).fbm(2, 0.5, 2.0);
    group.bench_function("simplex fbm 2 octaves", |b| {
        b.iter(|| noise.generate_3d(0.0, 0.0, 0.0, SIZE_3D, SIZE_3D, SIZE_3D))
    });

    let noise = Noise::simplex(0.01).fbm(8, 0.5, 2.0);
    group.bench_function("simplex fbm 8 octaves", |b| {
        b.iter(|| noise.generate_3d(0.0, 0.0, 0.0, SIZE_3D, SIZE_3D, SIZE_3D))
    });

    group
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(5));
}

criterion_group!(benches, perlin_2d, simplex_2d, perlin_3d, simplex_3d);
criterion_main!(benches);
