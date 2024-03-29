use crate::fluid_state::{FluidState, cal_pressure_corrections, cal_div};
use crate::pixelgrid::PixelGrid;
use crate::fluid_macroscopic_features::add_line;

pub trait PressureSolver {
    fn solve(&self, fs: &mut FluidState, pg: &PixelGrid, nits: usize);
}

pub struct JacobiPressureSolver {
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

        let max_value = fs.divergence
            .iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        assert!(*max_value < 0.000001);
    }

    #[test]
    fn test_pressure_solver_complex() {

        let m = 16;
        let n = 16;
        let pg = PixelGrid::new(m, n);
        let mut fs = FluidState::new(pg.m, pg.n);
        let ps = JacobiPressureSolver {};
    
        add_line(
            2, 2, 2, 12,
            0.01, 0.0, &mut fs, &pg
        );
        add_line(
            3, 2, 3, 12,
            -0.01, 0.0, &mut fs, &pg
        );

        fs.momentum_step(&pg, 1.0);
        fs.cal_divergence(&pg);
        ps.solve(&mut fs, &pg, 20);
        fs.apply_corrections();
        fs.cal_divergence(&pg);
        let max_value = fs.divergence
            .iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        println!("The maximum div is: {}", max_value);
        assert!(*max_value < 0.00051);

    }
}
