use crate::fluid_state::{FluidState, cal_pressure_corrections};
use crate::pixelgrid::PixelGrid;


pub trait PressureSolver {
    fn solve(&self, fs: &mut FluidState, pg: &PixelGrid, nits: usize);
}


pub struct JacobiPressureSolver {}


impl PressureSolver for JacobiPressureSolver {
    fn solve(&self, fs: &mut FluidState, pg: &PixelGrid, nits: usize) {

        let mut liipii: f32;
        let lii: f32 = - (2.0/pg.dx) - (2.0/pg.dy); // for an equal spaced grid, never changes
        let mut lap: f32;
        let dx2 = pg.dx * pg.dx;
        let dy2 = pg.dy * pg.dy;
        for _k in 0..nits {
            for ak in pg.n..(pg.mn-pg.n) {
                    if fs.boundary[ak] == 0.0 {
                        fs.pressure[ak] = 0.0;
                        continue;
                    }
                    liipii = fs.pressure[ak] * lii;

                    lap = ( (fs.boundary_x[ak + 1] * (fs.pressure[ak + 1] - fs.pressure[ak])
                             - fs.boundary_x[ak] * (fs.pressure[ak] - fs.pressure[ak - 1])) / dx2 )
                            + ( (fs.boundary_y[ak + pg.n] * (fs.pressure[ak + pg.n] - fs.pressure[ak])
                             - fs.boundary_y[ak] * (fs.pressure[ak] - fs.pressure[ak - pg.n])) / dy2 );

                    fs.pressure[ak] = (fs.divergence[ak] - ( lap - liipii)) / lii // TODO: do do at end;
            }
        }
        cal_pressure_corrections(fs, pg);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid_macroscopic_features::add_line;
    use crate::fluid_state::{cal_div};

    #[test]
    fn test_pressure_solver_simple() {
        let pg = PixelGrid::new(6, 6);  
        let mut fs = FluidState::new(&pg);   
        let ps = JacobiPressureSolver{};

        fs.u[3 * pg.n + 3] = 1.0;
        pg.print_data(&fs.u);
        cal_div(&fs.u, &fs.v, &mut fs.divergence, &pg);
        println!("Divergence:");
        pg.print_data(&fs.divergence);
        ps.solve(&mut fs, &pg, 100);
        println!("New pressure:");
        pg.print_data(&fs.pressure);
        fs.apply_corrections();
        fs.cal_divergence(&pg);
        println!("New divergence:");
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
        let mut fs = FluidState::new(&pg);
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
