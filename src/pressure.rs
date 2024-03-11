use crate::fluid_state::{FluidState, cal_pressure_corrections, cal_div};
use crate::pixelgrid::PixelGrid;

trait PressureSolver {
    fn solve(&self, fs: &mut FluidState, pg: &PixelGrid, nits: usize);
}

struct JacobiPressureSolver {
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

        let mut liipii: f32;
        let lii: f32 = - (2.0/pg.dx) - (2.0/pg.dy); // for an equal spaced grid, never changes

        for _k in 0..nits {
            cal_lap(fs, pg);
            for ak in 0..fs.pressure.len() {
                liipii = fs.pressure[ak] * lii;
                fs.pressure[ak] = fs.boundary[ak] * (fs.divergence[ak] - ( fs.laplacian[ak] - liipii)) / lii // TODO: do do at end;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pressure_solver_simple() {
        let pg = PixelGrid::new(6, 6);  
        let mut fs = FluidState::new(pg.m, pg.n);   
        let ps = JacobiPressureSolver{};

        fs.u[3 * pg.n + 3] = 1.0;
        pg.print_data(&fs.u);
        cal_div(&fs.u, &fs.v, &mut fs.divergence, &pg);
        pg.print_data(&fs.divergence);
        ps.solve(&mut fs, &pg, 100);
        pg.print_data(&fs.divergence);
        pg.print_data(&fs.pressure);
        fs.apply_corrections();
        fs.cal_divergence(&pg);
        pg.print_data(&fs.u);
        pg.print_data(&fs.v);
        pg.print_data(&fs.divergence);

    }
}