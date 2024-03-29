use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;

pub fn add_wave(i: usize, j: usize, length: usize, u: f32, v: f32, fs: &mut FluidState, pg: &PixelGrid) {

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
