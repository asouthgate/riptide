use cfd2d::pixelgrid::PixelGrid;
use cfd2d::fluid_state::FluidState;
use cfd2d::momentum::cal_new_velocity_boundary_aware_no_diffusion;
use std::time::{Instant, Duration};

fn main() {
    let m = 1024;
    let n = 1024;
    let mn = m * n;
    let pg = PixelGrid::new(m, n);
    let mut fs = FluidState::new(pg.m, pg.n);
    let mut ak = 0;
    let iterations = 10;
    let start_time = Instant::now(); // Start the timer
    for it in 1..iterations {
        for i in 1..m {
            for j in 1..n {
                ak = i * n + j;
                cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
            }
        }
        fs.swap_vectors();
    }

    let end_time = Instant::now(); // End the timer

    let elapsed_time = end_time - start_time; // Calculate the elapsed time
    let avg_time_per_iteration = elapsed_time / iterations as u32; // Calculate average time per iteration

    println!("Total time elapsed: {:?}", elapsed_time);
    println!("Average time per iteration: {:?}", avg_time_per_iteration);
}
