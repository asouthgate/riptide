use crate::fluid_state::FluidState;
use crate::pixelgrid::PixelGrid;

trait PressureSolver {
    fn solve(&self, fs: &mut FluidState, pg: &PixelGrid, nits: usize);
}

struct JacobiPressureSolver {
}

#[inline] 
fn cal_dwds_left(
    ak: usize,
    w: &Vec<f32>,
    arrdelta: usize, // which direction to take the difference
    d: f32
) -> f32 {
    (w[ak] - w[ak - arrdelta]) / d
}

#[inline] 
fn cal_dwds_right(
    ak: usize,
    w: &Vec<f32>,
    arrdelta: usize, // which direction to take the difference
    d: f32
) -> f32 {
    (w[ak + arrdelta] - w[ak]) / d
}

fn cal_div(
    u: &Vec<f32>, 
    v: &Vec<f32>,
    result_div: &mut Vec<f32>,
    pg: &PixelGrid
) {
    for i in 0..pg.m {
        for j in 0..pg.n {
            let ak: usize = i * pg.n + j;
            let dudx = cal_dwds_right(ak, u, 1, pg.dx);
            let dvdy = cal_dwds_right(ak, v, pg.n, pg.dy);
            result_div[ak] =  dudx + dvdy;
        }
    }
}

fn cal_pressure_corrections(
    fs: &mut FluidState, 
    pg: &PixelGrid
) {
    for i in 0..pg.m {
        for j in 0..pg.n {
            let ak: usize = i * pg.n + j;

            let mut is_boundary_x = fs.boundary[ak] == 0.0;
            let mut is_boundary_y = fs.boundary[ak] == 0.0;

            if ak > 0 {
                is_boundary_x = is_boundary_x || (fs.boundary[ak-1] == 0.0);
            }
            if ak > pg.n {
                is_boundary_y = is_boundary_y || (fs.boundary[ak-pg.n] == 0.0);
            }
            if !is_boundary_x { 
                fs.dpdx[ak] = cal_dwds_left(ak, &fs.pressure, 1, pg.dx);
            }
            if !is_boundary_y { 
                fs.dpdy[ak] = cal_dwds_left(ak, &fs.pressure, pg.n, pg.dy);
            }
        }
    }
}

fn cal_lap(
    fs: &mut FluidState, 
    pg: &PixelGrid
) {
    cal_pressure_corrections(fs, pg);
    cal_div(&fs.dpdx, &fs.dpdy, &mut fs.laplacian, pg);
}   

impl PressureSolver for JacobiPressureSolver {
    fn solve(&self, fs: &mut FluidState, pg: &PixelGrid, nits: usize) {

        let mut Liipii: f32;
        let Lii: f32 = -(2.0/pg.dx) - (2.0/pg.dy); // for an equal spaced grid, never changes

        for k in 0..nits {
            cal_lap(fs, pg);
            for ak in 0..fs.pressure.len() {
                Liipii = fs.pressure[ak] * Lii;
                fs.pressure[ak] = fs.boundary[ak] * (fs.divergence[ak] - ( fs.laplacian[ak] - Liipii)) // / Lii, but we do at end;
            }
        }

        for ak in 0..fs.pressure.len() {
            fs.pressure[ak] = fs.pressure[ak] / Lii;
        }
    }
}