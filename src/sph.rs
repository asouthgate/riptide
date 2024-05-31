use crate::particle::{Particle};
use crate::particle_ecs::*;
use crate::pixelgrid::PixelGrid;
use crate::particle_index::*;
use crate::kernels::*;
use std::time::Instant;


const PI: f32 = 3.141592653589793;

fn cal_dist(x0: (f32, f32), x1: (f32, f32)) -> f32 {
    ( (x0.0 - x1.0).powi(2) + (x0.1 - x1.1).powi(2) ).powf(0.5)
}


/// Representation of macroscopic fluid constants for particles.
///
/// In general, these constants may either be:
///     - A single scalar for all fluid
///     - A tuple of f32
///     - A vector, one scalar for each particle type
///     - A matrix, one scalar for each pairwise combination of types
///
/// These structures are intentionally simple. We want easy conversion to GPU.
/// They could be even simpler, and represented as arrays.
pub struct ParticleConstants {
    pub rho0_vec: Vec<f32>, // target density; hydrostatic force is zero at rho0
    pub c2_vec: Vec<f32>, // speed of sound squared
    pub mu_mat: Vec<Vec<f32>>, // viscosity
    pub s_mat: Vec<Vec<f32>>, // surface tension
    pub body_force: (f32, f32), // e.g. gravity
    pub gamma: f32 // exponential for pressure
}


fn _debug_print(p: &Particle) {
    println!("x: {:?} a: {:?} mass: {:?} density {:?} body: {:?} drag: {:?} hydro: {:?} surface: {:?}", 
        p.position, p.acceleration, p.mass, p.density, p.f_body, p.f_drag, p.f_hydro, p.f_surface
    );
}


fn _debug_print_ecs(p: ParticleRef) {
    println!("x: {:?} a: {:?} mass: {:?} density {:?} body: {:?} drag: {:?} hydro: {:?} surface: {:?}", 
        p.x, p.a, p.mass, p.density, p.f_body, p.f_viscous, p.f_pressure, p.f_surface
    );
}


/// Calculate the hydrostatic pressure force for a given density.
///
/// This model is essentially a spring
///
/// # Arguments
///
/// * `rho` - Density
/// * `rho0` - Density at rest
/// * `k` - Either speed of sound squared, or something else
#[allow(dead_code)]
fn cal_pressure(rho: f32, rho0: f32, k: f32) -> f32 {
    k * (rho - rho0)
}

fn cal_pressure_wcsph(rho: f32, rho0: f32, c2: f32, gamma: f32) -> f32 {
    // TODO: inefficient; bweak is const
    let bweak = c2 * rho0 / gamma;
    let result = bweak * ((rho/rho0).powf(gamma) - 1.0);
    result
}

/// Calculate the jth density contribution for particle i.
fn cal_rho_ij(mass_j: f32, dist_ij: f32, h: f32) -> f32 {
    mass_j * debrun_spiky_kernel(dist_ij, h)
}


fn _cal_pressure_force_coefficient(pi: f32, pj: f32, rhoi: f32, rhoj: f32, mj: f32) -> f32 {
    ( (pi/rhoi.powi(2)) + (pj/rhoj.powi(2)) ) * mj
}


/// Calculate the jth pressure force contribution for particle i.
//
/// * `pi` - pressure i
/// * `pj` - pressure j
/// * `rhoi` - density i
/// * `rhoj` - density j
/// * `mj` - mass j
/// * `gradW` - gradient of the kernel function W(r, h)
fn cal_pressure_force_ij(pi: f32, pj: f32, rhoi: f32, rhoj: f32, mj: f32, gradw: (f32, f32)) -> (f32, f32) {
    let pforce_coefficient = - pi * _cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, mj);
    (pforce_coefficient * gradw.0, pforce_coefficient * gradw.1)
}

// Update particle densities.
//
// This function takes a vector of particles, and for a given particle
// recomputes the density based on neighboring particles.
//
/// * `particle_data`
/// * `pindex`
/// * `h` - characteristic length
pub fn update_densities_ecs(
    x: &Vec<(f32, f32)>,
    mass: &Vec<f32>,
    density: &mut Vec<f32>,
    pindex: &ParticleIndex,
    n_particles: usize,
    h: f32,
    pg: &PixelGrid,
    nthread: usize
) {

    let n_chunks = nthread.min(n_particles);
    let chunk_size = n_particles / n_chunks;


    let _ = crossbeam::scope(|s| {
        for (i, density) in 
            density[0..n_particles].chunks_mut(chunk_size)
            .enumerate() 
        {
            let start = i * chunk_size;
            s.spawn(move |_| {
                for chunk_i in 0..density.len() { // ignore static particle
                    let i = chunk_i + start;
                    assert!(!x[i].0.is_nan());
                    assert!(!x[i].1.is_nan());            
                    density[chunk_i] = 0.0;
                    let slices = pindex.get_nbrs_nine_slice(&pg, x[i].0, x[i].1);
                    for slice in slices.iter() {
                        for &j in *slice {
                            let rij = cal_dist(x[i], x[j]);
                            let contrib = cal_rho_ij(mass[j], rij, h);
                            density[chunk_i] += contrib; 
                            assert!(!density[chunk_i].is_nan());
                        }            
                    }
                    assert!(density[chunk_i] > 0.0);
                }
            });
        }
    });
}


// Update particle pressure forces.
//
/// * `particle_data`
/// * `pindex`
/// * `h` - characteristic length
pub fn update_forces_ecs(
    x: &Vec<(f32, f32)>, 
    v: &Vec<(f32, f32)>,
    f_pressure: &mut Vec<(f32, f32)>, 
    f_viscous: &mut Vec<(f32, f32)>,
    f_surface: &mut Vec<(f32, f32)>,
    f_body: &mut Vec<(f32, f32)>,
    pressure: &Vec<f32>, 
    density: &Vec<f32>, 
    mass: &Vec<f32>, 
    particle_type: &Vec<usize>,
    pindex: &ParticleIndex,
    n_fluid_particles: usize,
    h: f32,
    mu_mat: &Vec<Vec<f32>>,
    s_mat: &Vec<Vec<f32>>,
    body_force: (f32, f32),
    pg: &PixelGrid,
    nthread: usize
) {

    let n_chunks = nthread.min(n_fluid_particles);
    let chunk_size = n_fluid_particles / n_chunks;

    crossbeam::scope(|s| {
        for (i, (((f_pressure, f_viscous), f_surface), f_body)) in 
            f_pressure[0..n_fluid_particles].chunks_mut(chunk_size)
            .zip(f_viscous[0..n_fluid_particles].chunks_mut(chunk_size))
            .zip(f_surface[0..n_fluid_particles].chunks_mut(chunk_size))
            .zip(f_body[0..n_fluid_particles].chunks_mut(chunk_size))
            .enumerate() 
        {
            let start = i * chunk_size;

            s.spawn(move |_| {
                let c_s = (3.0 * PI) / (2.0 * h);

                for chunk_i in 0..f_pressure.len() { // ignore static particle
                    let i = chunk_i + start;
                    f_body[chunk_i] = (body_force.0 * density[i], body_force.1 * density[i]);
                    assert!(!x[i].0.is_nan());
                    assert!(!x[i].1.is_nan());
                    assert!(!f_pressure[chunk_i].0.is_nan());
                    assert!(!f_pressure[chunk_i].1.is_nan());
                    f_pressure[chunk_i] = (0.0, 0.0);
                    f_viscous[chunk_i] = (0.0, 0.0);
                    f_surface[chunk_i] = (0.0, 0.0);
                    let mut ftot = (0.0, 0.0);
                    let mut f_viscous_tot = (0.0, 0.0);
                    let mut f_surface_tot = (0.0, 0.0);

                    let slices = pindex.get_nbrs_nine_slice(&pg, x[i].0, x[i].1);
                    for slice in slices.iter() {
                        for &nbrj in *slice {
                            if nbrj == i {
                                continue;
                            }
                            let dx = x[i].0 - x[nbrj].0;
                            let dy = x[i].1 - x[nbrj].1;
                            assert!(!dx.is_nan());
                            assert!(!dy.is_nan());
                            assert!(dx.powi(2) + dy.powi(2) > 0.0);
                            let grad = debrun_spiky_kernel_grad(dx, dy, h);
                            assert!(!grad.0.is_nan());
                            assert!(!grad.1.is_nan());
                            assert!(density[i] != 0.0);
                            assert!(density[nbrj] != 0.0);
                            let fij = cal_pressure_force_ij(
                                pressure[i],
                                pressure[nbrj],
                                density[i],
                                density[nbrj],
                                mass[nbrj],
                                grad
                            );
                            ftot.0 += fij.0;
                            ftot.1 += fij.1;
                            assert!(!fij.0.is_nan());
                            assert!(!fij.1.is_nan());
                            let r2 = dx.powi(2) + dy.powi(2);
                            let r = r2.powf(0.5);
                            let du = v[i].0 - v[nbrj].0;
                            let dv = v[i].1 - v[nbrj].1;
                            let muij = mu_mat[particle_type[i]][particle_type[nbrj]];
                            let a = 4.0 * mass[nbrj] / (density[nbrj] * density[i]);
                            let b = (grad.0 * du) + (grad.1 * dv);
                            let c = (dx / r2, dy / r2);

                            if r < h {
                                let s = s_mat[particle_type[i]][particle_type[nbrj]];
                                f_surface_tot.0 += s * (c_s * r).cos() * dx / r;
                                f_surface_tot.1 += s * (c_s * r).cos() * dy / r;
                            }

                            f_viscous_tot.0 += muij * a * b * c.0;
                            f_viscous_tot.1 += muij * a * b * c.1;
                        }
                    }
                    f_pressure[chunk_i] = ftot;
                    f_viscous[chunk_i] = f_viscous_tot;
                    f_surface[chunk_i] = f_surface_tot;
                }
            });
        }
    }).unwrap();
}


pub fn update_pressures_ecs(
    pressure: &mut Vec<f32>,
    density: &Vec<f32>,
    particle_type: &Vec<usize>,
    n_particles: usize,
    rho0_vec: &Vec<f32>, 
    c2_vec: &Vec<f32>,
) {
    for k in 0..n_particles {
        let pk = particle_type[k];
        // pressure[k] = cal_pressure(density[k], rho0_vec[pk], c2_vec[pk]);
        pressure[k] = cal_pressure_wcsph(density[k], rho0_vec[pk], c2_vec[pk], 7.0);
    }
}


pub fn update_body_forces_ecs(
    f_body: &mut Vec<(f32, f32)>,
    density: &Vec<f32>,
    n_fluid_particles: usize,
    body_force: (f32, f32)
) {
    for k in 0..n_fluid_particles {
        f_body[k] = (body_force.0 * density[k], body_force.1 * density[k]);
    }
}


pub fn leapfrog_update_acceleration_ecs(
    pdata: &mut ParticleData
) {
    for k in 0..pdata.n_fluid_particles {
        let ftotx = 
              pdata.f_pressure[k].0 
            + pdata.f_body[k].0
            + pdata.f_surface[k].0
            + pdata.f_viscous[k].0;

        let ftoty = 
            pdata.f_pressure[k].1 
            + pdata.f_body[k].1
            + pdata.f_surface[k].1
            + pdata.f_viscous[k].1;

        pdata.a[k] = (
            (1.0 / pdata.density[k]) * ftotx,
            (1.0 / pdata.density[k]) * ftoty
        );
    }
}


pub fn leapfrog_cal_forces_ecs(
    pg: &PixelGrid, pindex: &mut ParticleIndex,
    pdata: &ParticleData,
    pdata_new: &mut ParticleData,
    particle_constants: &ParticleConstants,
    h: f32,
    n_threads: usize
) {

    let t0 = Instant::now();

    pindex.update(pg, &pdata_new.x); // should be new
    let t1 = Instant::now();

    // update forces
    update_densities_ecs(
        &pdata_new.x, &pdata.mass, &mut pdata_new.density, pindex, pdata.n_particles, h, pg, n_threads
    );
    update_pressures_ecs(
        &mut pdata_new.pressure,
        &pdata_new.density,
        &pdata.particle_type,
        pdata.n_particles,
        &particle_constants.rho0_vec, 
        &particle_constants.c2_vec,
    );
    let t2 = Instant::now();

    update_forces_ecs(
        &pdata_new.x,
        &pdata_new.v,
        &mut pdata_new.f_pressure,
        &mut pdata_new.f_viscous,
        &mut pdata_new.f_surface,
        &mut pdata_new.f_body,
        &pdata_new.pressure, 
        &pdata_new.density,
        &pdata.mass,
        &pdata.particle_type,
        pindex,
        pdata.n_fluid_particles,
        h, 
        &particle_constants.mu_mat,
        &particle_constants.s_mat,
        particle_constants.body_force,
        pg,
        n_threads
    );
    let t3 = Instant::now();
    println!("\t\t cal forces: {:?}, density updates {:?}, pindex_update {:?}", t3-t2, t2-t1, t1-t0);

}

pub fn cal_dt(safety: f32, viscous_safety: f32, h: f32, cmax: f32, vmax: f32, mumax: f32) -> f32 {
    let a = safety * h / (vmax + cmax);
    let b = viscous_safety * h.powi(2) / mumax;
    a.min(b)
}

/// Compute a delta t by considering v and a of every particle
///
/// For a dt^2 + vdt = dx < h, we have to use the quadratic equation
/// With some care with absolute values and zeros
pub fn cal_dt_exhaustive(pdata: &ParticleData, h: f32, safety: f32) -> f32 {
    let mut dt_new: f32 = 100000.0;
    for k in 0..pdata.n_fluid_particles {
        let vkx = pdata.v[k].0.abs();
        let vky = pdata.v[k].1.abs();
        let akx = pdata.a[k].0.abs();
        let aky = pdata.a[k].1.abs();
        let dtkx: f32;
        let dtky: f32;
        if akx > 0.00001 {
            dtkx = (- vkx + (vkx.powi(2) + 4.0 * akx * h).powf(0.5) ) / ( 2.0 * akx );
        } else {
            dtkx = h / vkx;
        }
        if aky > 0.00001 {
            dtky = ( - vky + (vky.powi(2) + 4.0 * aky * h).powf(0.5) ) / ( 2.0 * aky );
        } else {
            dtky = h / vky;
        }
        dt_new = dt_new.min(dtkx).min(dtky);
        assert!(dt_new > 0.0);
    }
    safety * dt_new
}


pub fn leapfrog_ecs(
    pg: &PixelGrid, index: &mut ParticleIndex,
    pdata: &ParticleData,
    pdata_new: &mut ParticleData,
    particle_constants: &ParticleConstants, dt: f32,
    h: f32,
    n_threads: usize,
    dt_safety_factor: f32
) -> f32 {

    let t0 = Instant::now();
    for k in 0..pdata_new.n_fluid_particles {
        pdata_new.v[k] = (
            pdata.v[k].0 + pdata.a[k].0 * dt / 2.0,
            pdata.v[k].1 + pdata.a[k].1 * dt / 2.0
        );
    }
    let mut dt_new = cal_dt_exhaustive(pdata, h, dt_safety_factor);
    for k in 0..pdata_new.n_fluid_particles {
        pdata_new.x[k] = (
            pdata.x[k].0 + dt_new * pdata_new.v[k].0,
            pdata.x[k].1 + dt_new * pdata_new.v[k].1
        );
    }  
    let t1 = Instant::now();

    leapfrog_cal_forces_ecs(
        pg, index, pdata, pdata_new, particle_constants, h, n_threads
    );
    let t2 = Instant::now();

    leapfrog_update_acceleration_ecs(pdata_new);

    let t3 = Instant::now();
    dt_new = cal_dt_exhaustive(pdata_new, h, dt_safety_factor);
    assert!(dt_new < 1000000.0);

    let t4 = Instant::now();

    for k in 0..pdata_new.n_fluid_particles {
        pdata_new.v[k] = (
            pdata_new.v[k].0 + pdata_new.a[k].0 * dt_new / 2.0,
            pdata_new.v[k].1 + pdata_new.a[k].1 * dt_new / 2.0
        );
    } 
    let t5 = Instant::now();
    println!("\t v1: {:?}, cal_dt {:?}, acc {:?} forces {:?}, v1/2 {:?}", t5-t4, t4-t3, t3-t2, t2-t1, t1-t0);
    dt_new = cal_dt_exhaustive(pdata_new, h, dt_safety_factor);
    dt_new
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_cal_p() {
        assert!(cal_pressure(1.0, 0.0, 1.0) == 1.0);
        assert!(cal_pressure(1.0, 0.0, 2.0) == 2.0);
    }

    #[test]
    fn test_cal_pressure_force_coeff_ij() {
        let rhoi = 1.0;
        let rhoj = 2.0;
        let pi = cal_pressure(rhoi, 0.0, 2.0); 
        let pj = cal_pressure(rhoj, 0.0, 2.0);
        let m = 1.0;
        assert!(pi == 2.0);
        assert!(pj == 4.0);
        println!("{} {}", _cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, m), (2 + 1));
        assert!(_cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, m) == (2.0 + 1.0));
    }

    #[test]
    fn test_2_particles() {
        let h: f32 = 2.0;
        let mut dt: f32;
        let mut pdata = ParticleData::new(2, 2);
        pdata.x[0] = (10.0, 10.0);
        pdata.x[1] = (10.5, 10.0);
        let pg = PixelGrid::new(1000, 1000);
        let mut index = ParticleIndex::new(&pg, 2); 
        index.update(&pg, &pdata.x);
        let mut prev_err = 99999.0;

        let pc = ParticleConstants {
            rho0_vec: vec![0.9, 0.9],
            c2_vec: vec![3.4, 3.8],
            mu_mat: vec![
                vec![2.5, 0.01], 
                vec![0.01, 3.0]
            ],
            s_mat: vec![
                vec![1.0, 0.0], 
                vec![0.0, 20.0]
            ],
            body_force: (0.0, -0.9),
            gamma: 7.0
        };

        let mut pdata_new = pdata.clone();
        assert!(pdata_new.density == pdata.density);

        for pi in 0..pdata.n_particles {
            _debug_print_ecs(pdata.get_particle_ref(pi));   
            _debug_print_ecs(pdata_new.get_particle_ref(pi));   
        }
        println!("");

        leapfrog_cal_forces_ecs(
            &pg, &mut index,
            &pdata,
            &mut pdata_new,
            &pc, h, 2
        );
        leapfrog_update_acceleration_ecs(&mut pdata_new);
        dt = cal_dt_exhaustive(&pdata_new, h, 0.01);

        println!("INIT LOOP");

        for pi in 0..pdata.n_particles {
            println!("{}", pi);
            _debug_print_ecs(pdata.get_particle_ref(pi));   
            _debug_print_ecs(pdata_new.get_particle_ref(pi));   
        }
        mem::swap(&mut pdata, &mut pdata_new);

        for _ in 0..20 {
            
            dt = leapfrog_ecs(
                &pg, &mut index,
                &pdata, &mut pdata_new,
                &pc, dt, h, 5, 0.1
            );  
            
            let mut new_err = 0.0;

            for pi in 0..pdata.n_particles {
                println!("{}", pi);
                _debug_print_ecs(pdata.get_particle_ref(pi));   
                _debug_print_ecs(pdata_new.get_particle_ref(pi));   
            }

            for pi in 0..pdata_new.n_fluid_particles {
                let rho0 = pc.rho0_vec[pdata_new.particle_type[pi]];
                new_err += (pdata_new.density[pi] - rho0).abs()
            }    
            assert!(new_err <= prev_err);

            prev_err = new_err; 

            mem::swap(&mut pdata, &mut pdata_new);

        }
    }

}
