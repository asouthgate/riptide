use riptide::pixelgrid::PixelGrid;
use riptide::fluid_state::FluidState;
use riptide::pressure::{JacobiPressureSolver, PressureSolver};
use riptide::momentum::cal_new_velocity_boundary_aware_no_diffusion;
use std::time::{Duration, Instant};

fn main() {
    let m = 256;
    let n = 256;
    let pg = PixelGrid::new(m, n);
    let mut fs = FluidState::new(&pg);
    let ps = JacobiPressureSolver {};
    let iterations = 10;
    let pressure_iterations = 5;
    let start_time = Instant::now();
    let mut momentum_time = Duration::new(0, 0);
    let mut cal_divergence_time = Duration::new(0, 0);
    let mut pressure_solve_time = Duration::new(0, 0);
    let mut apply_corrections_time = Duration::new(0, 0);
    for _it in 1..iterations {

        let t0 = Instant::now();
        fs.momentum_step(&pg, 1.0);
        let t1 = Instant::now();
        momentum_time += t1 - t0;

        fs.cal_divergence(&pg);
        let t2 = Instant::now();
        cal_divergence_time += t2 - t1;

        ps.solve(&mut fs, &pg, pressure_iterations);
        let t3 = Instant::now();
        pressure_solve_time += t3 - t2;

        fs.apply_corrections();
        let t4 = Instant::now();
        apply_corrections_time += t4 - t3;
    }

    let end_time = Instant::now();

    let elapsed_time = end_time - start_time;
    let avg_time_per_iteration = elapsed_time / iterations as u32;

    println!("Total time elapsed: {:?}", elapsed_time);
    println!("Average time per iteration: {:?}", avg_time_per_iteration);
    println!("Average function times: 
        momentum_time: {:?},
        cal_divergence_time: {:?},
        pressure_solve_time: {:?},
        pressure_solve_time per iteration: {:?},
        apply_corrections_time: {:?}
    ", 
    momentum_time / iterations as u32,
    cal_divergence_time / iterations as u32, 
    pressure_solve_time / iterations as u32, 
    (pressure_solve_time / pressure_iterations as u32) / iterations as u32,
    apply_corrections_time / iterations as u32
    )
}
