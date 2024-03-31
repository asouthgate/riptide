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
    let mut ak;
    let iterations = 10;
    let start_time = Instant::now(); // Start the timer
    for _it in 1..iterations {
        for i in 1..m {
            for j in 1..n {
                ak = i * n + j;
                cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
            }
        }
        ps.solve(&mut fs, &pg, 5);
        fs.swap_vectors();
    }

    let end_time = Instant::now(); // End the timer

    let elapsed_time = end_time - start_time; // Calculate the elapsed time
    let avg_time_per_iteration = elapsed_time / iterations as u32; // Calculate average time per iteration

    println!("Total time elapsed: {:?}", elapsed_time);
    println!("Average time per iteration: {:?}", avg_time_per_iteration);
}
