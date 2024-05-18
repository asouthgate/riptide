use riptide::pixelgrid::PixelGrid;
use riptide::fluid_state::FluidState;
use riptide::pressure::{JacobiPressureSolver, PressureSolver};
use std::time::{Duration, Instant};

fn main() {
    let m = 144;
    let n = 256;
    let pg = PixelGrid::new(m, n);
    let mut fs = FluidState::new(&pg);
    let ps = JacobiPressureSolver {};

    // let rho0 = 1.0;
    // let c0 = 0.1;
    // let dt = 1.0;
    // let h = 1.0;
    // let mut particles: Vec<Particle> = vec![];
    // for i in (1..pg.m-1).step_by(32) {
    //     for j in (1..pg.n-1).step_by(32) {
    //         particles.push(Particle{
    //             position: (j as f32 + pg.dx * 0.5, i as f32 + pg.dy * 0.5), mass: 1.0,
    //             .. Default::default()
    //         });
    //     }
    // }
    // let n_real_particles = particles.len();
    // let mut particle_index = ParticleIndex::new(&pg);
    // particles.extend(get_ghost_particles_naive(&fs, &pg));


    let iterations = 10;
    let pressure_iterations = 2;
    let start_time = Instant::now();
    let mut momentum_time = Duration::new(0, 0);
    let mut limit_and_cool_time = Duration::new(0, 0);
    let mut cal_divergence_time = Duration::new(0, 0);
    let mut pressure_solve_time = Duration::new(0, 0);
    let mut apply_corrections_time = Duration::new(0, 0);
    let mut sph_time = Duration::new(0, 0);
    for _it in 1..iterations {

        let t0 = Instant::now();
        fs.momentum_step(&pg, 1.0);
        let t1 = Instant::now();
        momentum_time += t1 - t0;

        fs.limit(&pg, 1.0);
        fs.cool(&pg, 0.1, 0.1, 1.0);
        let t2 = Instant::now();
        limit_and_cool_time += t2 - t1;

        fs.cal_divergence(&pg);
        let t3 = Instant::now();
        cal_divergence_time += t3 - t2;

        ps.solve(&mut fs, &pg, pressure_iterations);
        let t4 = Instant::now();
        pressure_solve_time += t4 - t3;

        fs.apply_corrections();
        let t5 = Instant::now();
        apply_corrections_time += t5 - t4;

        // update_all_particles(
        //     &pg, &fs, &mut particle_index,
        //     &mut particles, n_real_particles,
        //     rho0, c0, h, dt, (0.0, 0.0), 0.0
        // );
        let t6 = Instant::now();
        sph_time = t6-t5;
        
    }

    let end_time = Instant::now();

    let elapsed_time = end_time - start_time;
    let avg_time_per_iteration = elapsed_time / iterations as u32;

    println!("Total time elapsed: {:?}", elapsed_time);
    println!("Average time per iteration: {:?}", avg_time_per_iteration);
    println!("Average function times: 
        momentum_time: {:?},
        limit_and_cool_time: {:?},
        cal_divergence_time: {:?},
        pressure_solve_time: {:?},
        pressure_solve_time per iteration: {:?},
        apply_corrections_time: {:?},
        sph_time: {:?}
    ", 
    momentum_time / iterations as u32,
    limit_and_cool_time / iterations as u32,
    cal_divergence_time / iterations as u32, 
    pressure_solve_time / iterations as u32, 
    (pressure_solve_time / pressure_iterations as u32) / iterations as u32,
    apply_corrections_time / iterations as u32,
    sph_time / iterations as u32
    )
}
