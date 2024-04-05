use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;
use rand::prelude::*;

// Add wavey noise to random pixels.
//
// For half of the total velocity specified,
// apportion uniformly. For the other half,
// apportion via a random stick-breaking process.
pub fn add_random_wavey_noise(
    total_velocity_x: f32,
    total_velocity_y: f32,
    n_points: usize,
    fs: &mut FluidState,
    pg: &PixelGrid
) {
    let mut rng = rand::thread_rng();
    let mut remaining_u = total_velocity_x / 2.0;
    let mut remaining_v = total_velocity_y / 2.0;
    for _pt in 0..n_points {
        let rand_i = rng.gen_range(0..pg.m as i32) as usize;
        let rand_j = rng.gen_range(0..pg.n as i32) as usize;
        let ak = rand_i * pg.n + rand_j;
        fs.u[ak] += total_velocity_x * 0.5 / n_points as f32;
        fs.v[ak] += total_velocity_y * 0.5 / n_points as f32;
        let du = rng.gen_range(0.0..remaining_u);
        let dv = rng.gen_range(0.0..remaining_v);
        remaining_u -= du;
        remaining_v -= dv;
        fs.u[ak] += du;
        fs.v[ak] += dv;
    }
}


pub fn add_wave(
    i: usize,
    j: usize,
    length: usize,
    u: f32,
    v: f32,
    fs: &mut FluidState,
    pg: &PixelGrid
) {
    let norm = ( u.powf(2.0) + v.powf(2.0) ).powf(0.5);
    let vnorm = v/norm;
    let unorm = u/norm;

    let xdir = -vnorm;
    let ydir = unorm;

    // required because root 2 guaranteed to step a whole pixel
    // but can't step two pixels at once
    let sqrt2 = (2.0 as f32).powf(0.5); 

    for l in 0..length {        
        let di = ydir * sqrt2 * l as f32;
        let dj = xdir * sqrt2 * l as f32;
        let dif = di.floor() as i32;
        let djf = dj.floor() as i32;
        let i2 = (i as i32 + dif ) as usize;
        let j2 = (j as i32 + djf ) as usize;

        let ak = i2 * pg.n + j2;
        fs.u[ak] += u;
        fs.v[ak] += v;
    }
}

pub fn add_line(i0: usize, j0: usize, ie: usize, je: usize, u: f32, v: f32, fs: &mut FluidState, pg: &PixelGrid) {
    let sqrt2 = (2.0 as f32).powf(0.5);
    let xdiff = (je-j0) as f32;
    let ydiff = (ie-i0) as f32;
    let length = (xdiff.powf(2.0) + ydiff.powf(2.0)).powf(0.5);
    let xdir = xdiff / length;
    let ydir = ydiff / length;

    for l in 0..length.round() as usize {
        let di = ydir * sqrt2 * l as f32;
        let dj = xdir * sqrt2 * l as f32;
        let dif = di.floor() as i32;
        let djf = dj.floor() as i32;
        let i2 = (i0 as i32 + dif ) as usize;
        let j2 = (j0 as i32 + djf ) as usize;

        let ak = i2 * pg.n + j2;
        fs.u[ak] += u;
        fs.v[ak] += v;
    }

}
