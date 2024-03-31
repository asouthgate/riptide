use riptide::pixelgrid::PixelGrid;
use riptide::fluid_state::FluidState;
use riptide::pressure::{JacobiPressureSolver, PressureSolver};
use riptide::momentum::cal_new_velocity_boundary_aware_no_diffusion;
use std::time::{Instant};

fn main() {
    let m = 512;
    let n = 512;
    let pg = PixelGrid::new(m, n);
    let mut fs = FluidState::new(pg.m, pg.n);
    let ps = JacobiPressureSolver {};
    let iterations = 10;
    let start_time = Instant::now();
    for _it in 1..iterations {
        fs.momentum_step(&pg, 1.0);
        fs.cal_divergence(&pg);
        ps.solve(&mut fs, &pg, 5);
        fs.apply_corrections();
    }

    let end_time = Instant::now();

    let elapsed_time = end_time - start_time; // Calculate the elapsed time
    let avg_time_per_iteration = elapsed_time / iterations as u32; // Calculate average time per iteration

    println!("Total time elapsed: {:?}", elapsed_time);
    println!("Average time per iteration: {:?}", avg_time_per_iteration);
}
