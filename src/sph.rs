use crate::particle::{Particle};
use crate::particle_ecs::*;
use crate::pixelgrid::PixelGrid;
use crate::particle_index::*;
use crate::kernels::*;
use std::time::Instant;
use std::mem;


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
fn cal_pressure(rho: f32, rho0: f32, k: f32) -> f32 {
    k * (rho - rho0)
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
    neighbors: &Vec<Vec<usize>>,
    n_particles: usize,
    h: f32
) {
    for i in 0..n_particles {
        assert!(!x[i].0.is_nan());
        assert!(!x[i].1.is_nan());
        density[i] = 0.0;
        for j in neighbors[i].iter() {
            let rij = cal_dist(x[i], x[*j]);
            let contrib = cal_rho_ij(mass[*j], rij, h);
            density[i] += contrib; 
            assert!(!density[i].is_nan());
        }
    }
}

// Update particle densities.
//
// This function takes a vector of particles, and for a given particle
// recomputes the density based on neighboring particles.
//
/// * `particles` - vector of particles
/// * `h` - characteristic length
pub fn update_densities(
    particles: &mut Vec<Particle>, h: f32
) {
    for i in 0..particles.len() {
        assert!(!particles[i].get_x().is_nan());
        assert!(!particles[i].get_y().is_nan());
        particles[i].density = 0.0;
        let mut density_sum = 0.0;
        for nbrj in particles[i].nbrs.iter() {
            let rij = particles[i].dist(&particles[*nbrj]);
            let contrib = cal_rho_ij(particles[*nbrj].mass, rij, h);
            density_sum += contrib; 
            assert!(!density_sum.is_nan());
        }
        particles[i].density = density_sum;
    }
}

// Update particle pressure forces.
//
/// * `particle_data`
/// * `pindex`
/// * `h` - characteristic length
pub fn update_pressure_forces_ecs(
    x: &Vec<(f32, f32)>, 
    f_pressure: &mut Vec<(f32, f32)>, 
    pressure: &Vec<f32>, 
    density: &Vec<f32>, 
    mass: &Vec<f32>, 
    neighbors: &Vec<Vec<usize>>,
    n_fluid_particles: usize,
    h: f32
) {
    for i in 0..n_fluid_particles { // ignore static particles
        assert!(!x[i].0.is_nan());
        assert!(!x[i].1.is_nan());
        f_pressure[i] = (0.0, 0.0);
        let mut ftot = (0.0, 0.0);
        for _nbrj in neighbors[i].iter() {
            let nbrj = *_nbrj;
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
        }
        f_pressure[i] = ftot;
        assert!(!f_pressure[i].0.is_nan());
        assert!(!f_pressure[i].1.is_nan());
    }
}

// Update particle pressure forces.
//
/// * `particles` - vector of particles
/// * `h` - characteristic length
/// * `n_real_particles` - gives index of last particle to compute for
///         e.g. to ignore static particles
pub fn update_pressure_forces(
    particles: &mut Vec<Particle>, h: f32, n_real_particles: usize
) {
    for i in 0..n_real_particles {
        let x = particles[i].get_x();
        let y = particles[i].get_y();
        assert!(!x.is_nan());
        assert!(!y.is_nan());
        particles[i].f_hydro = (0.0, 0.0);
        let mut ftot = (0.0, 0.0);
        for _nbrj in particles[i].nbrs.iter() {
            let nbrj = *_nbrj;
            if nbrj == i {
                continue;
            }
            let dx = particles[i].get_x() - particles[nbrj].get_x();
            let dy = particles[i].get_y() - particles[nbrj].get_y();
            assert!(!dx.is_nan());
            assert!(!dy.is_nan());
            assert!(dx.powi(2) + dy.powi(2) > 0.0);
            let grad = debrun_spiky_kernel_grad(dx, dy, h);
            assert!(!grad.0.is_nan());
            assert!(!grad.1.is_nan());
            assert!(particles[i].density != 0.0);
            assert!(particles[nbrj].density != 0.0);
            let fij = cal_pressure_force_ij(
                particles[i].pressure,
                particles[nbrj].pressure,
                particles[i].density,
                particles[nbrj].density,
                particles[nbrj].mass,
                grad
            );
            ftot.0 += fij.0;
            ftot.1 += fij.1;
            assert!(!fij.0.is_nan());
            assert!(!fij.1.is_nan());
        }
        particles[i].f_hydro = ftot;
        assert!(!particles[i].f_hydro.0.is_nan());
        assert!(!particles[i].f_hydro.1.is_nan());
    }
}

pub fn update_pressures_ecs(
    pressure: &mut Vec<f32>,
    density: &Vec<f32>,
    particle_type: &Vec<usize>,
    n_particles: usize,
    rho0_vec: &Vec<f32>, 
    c2_vec: &Vec<f32>
) {
    for k in 0..n_particles {
        let pk = particle_type[k];
        pressure[k] = cal_pressure(density[k], rho0_vec[pk], c2_vec[pk]);
    }
}

pub fn update_pressures(particles: &mut Vec<Particle>, rho0_vec: &Vec<f32>, c2_vec: &Vec<f32>) {
    for k in 0..particles.len() {
        let pk = particles[k].particle_type;
        particles[k].pressure = cal_pressure(particles[k].density, rho0_vec[pk], c2_vec[pk]);
    }
}


pub fn update_body_forces(particles: &mut Vec<Particle>, n_real_particles: usize, body_force: (f32, f32)) {
    for k in 0..n_real_particles {
        particles[k].f_body = (body_force.0 * particles[k].density, body_force.1 * particles[k].density);
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


// Update particle surface tension forces.
//
/// * `particles` - vector of particles
/// * `h` - characteristic length
/// * `n_real_particles` - gives index of last particle to compute for
///         e.g. to ignore static particles
/// * `s_mat` - pairwise surface tension constants
pub fn update_surface_forces(
    particles: &mut Vec<Particle>, 
    n_real_particles: usize, 
    h: f32, 
    s_mat: &Vec<Vec<f32>>
) {
    let c_s = (3.0 * PI) / (2.0 * h);
    for i in 0..n_real_particles {
        let x = particles[i].get_x();
        let y = particles[i].get_y();
        particles[i].f_surface = (0.0, 0.0);
        assert!(!x.is_nan());
        assert!(!y.is_nan());
        let mut f_surface_tot = (0.0, 0.0);
        for _nbrj in particles[i].nbrs.iter() {
            let nbrj = *_nbrj;
            if nbrj == i {
                continue;
            }
            let dx = particles[i].get_x() - particles[nbrj].get_x();
            let dy = particles[i].get_y() - particles[nbrj].get_y();
            let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
            if r < h {
                let s = s_mat[particles[i].particle_type][particles[nbrj].particle_type];
                f_surface_tot.0 += s * (c_s * r).cos() * dx / r;
                f_surface_tot.1 += s * (c_s * r).cos() * dy / r;
            }
        }
        particles[i].f_surface = f_surface_tot;
    }
}


// Update particle surface tension forces.
//
/// * `particles` - vector of particles
/// * `h` - characteristic length
/// * `n_real_particles` - gives index of last particle to compute for
///         e.g. to ignore static particles
/// * `s_mat` - pairwise surface tension constants
pub fn update_surface_forces_ecs(
    x: &Vec<(f32, f32)>,
    f_surface: &mut Vec<(f32, f32)>,
    particle_type: &Vec<usize>,
    neighbors: &Vec<Vec<usize>>,
    n_fluid_particles: usize,
    h: f32, 
    s_mat: &Vec<Vec<f32>>
) {
    let c_s = (3.0 * PI) / (2.0 * h);
    for i in 0..n_fluid_particles {
        f_surface[i] = (0.0, 0.0);
        assert!(!x[i].0.is_nan());
        assert!(!x[i].1.is_nan());
        let mut f_surface_tot = (0.0, 0.0);
        for _nbrj in neighbors[i].iter() {
            let nbrj = *_nbrj;
            if nbrj == i {
                continue;
            }
            let dx = x[i].0 - x[nbrj].0;
            let dy = x[i].1 - x[nbrj].1;
            let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
            if r < h {
                let s = s_mat[particle_type[i]][particle_type[nbrj]];
                f_surface_tot.0 += s * (c_s * r).cos() * dx / r;
                f_surface_tot.1 += s * (c_s * r).cos() * dy / r;
            }
        }
        f_surface[i] = f_surface_tot;
    }
}


// Update particle viscous drag forces.
//
/// * `particles` - vector of particles
/// * `h` - characteristic length
/// * `n_real_particles` - gives index of last particle to compute for
///         e.g. to ignore static particles
/// * `mu_mat` - pairwise surface tension constants
pub fn update_viscous_forces_ecs(
    x: &Vec<(f32, f32)>,
    v: &Vec<(f32, f32)>,
    f_viscous: &mut Vec<(f32, f32)>,
    particle_type: &Vec<usize>,
    mass: &Vec<f32>,
    density: &Vec<f32>,
    neighbors: &Vec<Vec<usize>>,
    n_fluid_particles: usize,
    h: f32, 
    mu_mat: &Vec<Vec<f32>>
) {
    for i in 0..n_fluid_particles {
        f_viscous[i] = (0.0, 0.0);
        assert!(!x[i].0.is_nan());
        assert!(!x[i].1.is_nan());
        let mut f_viscous_tot = (0.0, 0.0);
        for _nbrj in neighbors[i].iter() {
            let nbrj = *_nbrj;
            if nbrj == i {
                continue;
            }
            let dx = x[i].0 - x[nbrj].0;
            let dy = x[i].1 - x[nbrj].1;
            let r2 = dx.powi(2) + dy.powi(2);
            let du = v[i].0 - v[nbrj].0;
            let dv = v[i].1 - v[nbrj].1;
            let grad = debrun_spiky_kernel_grad(dx, dy, h);
            let muij = mu_mat[particle_type[i]][particle_type[nbrj]];
            let a = 4.0 * mass[nbrj] / (density[nbrj] * density[i]);
            let b = (grad.0 * du) + (grad.1 * dv);
            let c = (dx / r2, dy / r2);
            f_viscous_tot.0 += muij * a * b * c.0;
            f_viscous_tot.1 += muij * a * b * c.1;

        }
        f_viscous[i] = f_viscous_tot;
    }
}


// Update particle viscous drag forces.
//
/// * `particles` - vector of particles
/// * `h` - characteristic length
/// * `n_real_particles` - gives index of last particle to compute for
///         e.g. to ignore static particles
/// * `mu_mat` - pairwise surface tension constants
pub fn update_viscous_forces(
    particles: &mut Vec<Particle>, 
    n_real_particles: usize, h: f32, mu_mat: &Vec<Vec<f32>>
) {
    for i in 0..n_real_particles {
        let x = particles[i].get_x();
        let y = particles[i].get_y();
        particles[i].f_drag = (0.0, 0.0);
        assert!(!x.is_nan());
        assert!(!y.is_nan());
        let mut f_drag_tot = (0.0, 0.0);
        for _nbrj in particles[i].nbrs.iter() {
            let nbrj = *_nbrj;
            if nbrj == i {
                continue;
            }
            let dx = particles[i].get_x() - particles[nbrj].get_x();
            let dy = particles[i].get_y() - particles[nbrj].get_y();
            let r2 = dx.powi(2) + dy.powi(2);
            let du = particles[i].get_u() - particles[nbrj].get_u();
            let dv = particles[i].get_v() - particles[nbrj].get_v();
            let grad = debrun_spiky_kernel_grad(dx, dy, h);
            let muij = mu_mat[particles[i].particle_type][particles[nbrj].particle_type];
            let a = 4.0 * particles[nbrj].mass / (particles[nbrj].density * particles[i].density);
            let b = (grad.0 * du) + (grad.1 * dv);
            let c = (dx / r2, dy / r2);
            f_drag_tot.0 += muij * a * b * c.0;
            f_drag_tot.1 += muij * a * b * c.1;

        }
        particles[i].f_drag = f_drag_tot;
    }
}


pub fn leapfrog_update_acceleration(particles: &mut Vec<Particle>, n_real_particles: usize) {
    for k in 0..n_real_particles {
        let ftotx = 
              particles[k].f_hydro.0 
            + particles[k].f_body.0
            + particles[k].f_surface.0
            + particles[k].f_drag.0;

        let ftoty = 
              particles[k].f_hydro.1
            + particles[k].f_body.1
            + particles[k].f_surface.1
            + particles[k].f_drag.1;

        particles[k].acceleration = (
            (1.0 / particles[k].density) * ftotx,
            (1.0 / particles[k].density) * ftoty
        );
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
    h: f32
) {
    let max_dist = h * 1.0;
    pindex.update_ecs(pg, pdata);
    pindex.update_neighbors(pg, pdata, max_dist);

    // update forces
    update_densities_ecs(
        &pdata_new.x, &pdata.mass, &mut pdata_new.density, &pindex.neighbors, pdata.n_particles, h
    );
    update_body_forces_ecs(
        &mut pdata_new.f_body, &pdata_new.density, pdata.n_fluid_particles, particle_constants.body_force
    );
    update_surface_forces_ecs(
        &pdata_new.x,
        &mut pdata_new.f_surface,
        &pdata.particle_type,
        &pindex.neighbors,
        pdata.n_fluid_particles,
        h,
        &particle_constants.s_mat
    );
    update_viscous_forces_ecs(
        &pdata_new.x,
        &pdata_new.v,
        &mut pdata_new.f_viscous,
        &pdata.particle_type,
        &pdata.mass,
        &pdata_new.density,
        &pindex.neighbors,
        pdata.n_fluid_particles,
        h, 
        &particle_constants.mu_mat
    );
    update_pressures_ecs(
        &mut pdata_new.pressure,
        &pdata_new.density,
        &pdata.particle_type,
        pdata.n_particles,
        &particle_constants.rho0_vec, 
        &particle_constants.c2_vec
    );
    update_pressure_forces_ecs(
        &pdata_new.x, 
        &mut pdata_new.f_pressure, 
        &pdata_new.pressure, 
        &pdata_new.density,
        &pdata.mass,
        &pindex.neighbors,
        pdata.n_fluid_particles,
        h    
    );
}


pub fn leapfrog_cal_forces(
    pg: &PixelGrid, index: &mut ParticleIndex,
    particles: &mut Vec<Particle>, n_real_particles: usize,
    particle_constants: &ParticleConstants,
    h: f32
) {
    let max_dist = h * 1.0;
    index.update(pg, particles);
    for i in 0..particles.len() {
        particles[i].nbrs = index.get_nbrs(&pg, particles[i].get_x(), particles[i].get_y(), max_dist);
        cull_nbrs(i, particles, h);
    }

    // update forces
    update_densities(particles, h);
    update_body_forces(particles, n_real_particles, particle_constants.body_force);
    update_surface_forces(particles, n_real_particles, h, &particle_constants.s_mat);
    update_viscous_forces(particles, n_real_particles, h, &particle_constants.mu_mat);
    update_pressures(particles, &particle_constants.rho0_vec, &particle_constants.c2_vec);
    update_pressure_forces(particles, h, n_real_particles);
}


pub fn leapfrog(
    pg: &PixelGrid, index: &mut ParticleIndex,
    particles: &mut Vec<Particle>, n_real_particles: usize,
    particle_constants: &ParticleConstants, dt: f32,
    h: f32
) {

    for k in 0..n_real_particles {
        particles[k].velocity = (
            particles[k].velocity.0 + particles[k].acceleration.0 * dt / 2.0,
            particles[k].velocity.1 + particles[k].acceleration.1 * dt / 2.0
        );
        particles[k].position = (
            particles[k].position.0 + dt * particles[k].velocity.0,
            particles[k].position.1 + dt * particles[k].velocity.1
        );
    }  
    leapfrog_cal_forces(
        pg, index, particles, n_real_particles, particle_constants, h
    );
    leapfrog_update_acceleration(particles, n_real_particles);
    for k in 0..n_real_particles {
        particles[k].velocity = (
            particles[k].velocity.0 + particles[k].acceleration.0 * dt / 2.0,
            particles[k].velocity.1 + particles[k].acceleration.1 * dt / 2.0
        );
    }  
}


pub fn leapfrog_ecs(
    pg: &PixelGrid, index: &mut ParticleIndex,
    pdata: &ParticleData,
    pdata_new: &mut ParticleData,
    particle_constants: &ParticleConstants, dt: f32,
    h: f32,
) {

    let nthread = 12;
    let n_chunks = nthread.min(pdata_new.n_fluid_particles);
    let chunk_size = pdata_new.n_fluid_particles / n_chunks;

    let t0 = Instant::now();
    println!("{} {} {} {}", nthread, n_chunks, chunk_size, pdata_new.n_fluid_particles);
    crossbeam::scope(|s| {
        for (i, ((((a_prev, v_new), v_prev), x_new), x_prev)) in 
            pdata.a.chunks(chunk_size)
            .zip(pdata_new.v.chunks_mut(chunk_size))
            .zip(pdata.v.chunks(chunk_size))
            .zip(pdata_new.x.chunks_mut(chunk_size))
            .zip(pdata.x.chunks(chunk_size))
            .enumerate() 
        {
            let start = i * chunk_size;
            s.spawn(move |_| {
                for k in start..v_prev.len() {
                    v_new[k] = (
                        v_prev[k].0 + a_prev[k].0 * dt / 2.0,
                        v_prev[k].1 + a_prev[k].1 * dt / 2.0
                    );
                    x_new[k] = (
                        x_prev[k].0 + dt * v_new[k].0,
                        x_prev[k].1 + dt * v_new[k].1
                    );

                }  
            });
        }
    }).unwrap();
    let t1 = Instant::now();

    for k in 0..pdata_new.n_fluid_particles {
        pdata_new.v[k] = (
            pdata.v[k].0 + pdata.a[k].0 * dt / 2.0,
            pdata.v[k].1 + pdata.a[k].1 * dt / 2.0
        );
        pdata_new.x[k] = (
            pdata.x[k].0 + dt * pdata_new.v[k].0,
            pdata.x[k].1 + dt * pdata_new.v[k].1
        );
    }  
    let t2 = Instant::now();
    println!("{:?} {:?}", t1-t0, t2-t1);

    leapfrog_cal_forces_ecs(
        pg, index, pdata, pdata_new, particle_constants, h
    );
    leapfrog_update_acceleration_ecs(pdata_new);
    
    for k in 0..pdata_new.n_fluid_particles {
        pdata_new.v[k] = (
            pdata_new.v[k].0 + pdata_new.a[k].0 * dt / 2.0,
            pdata_new.v[k].1 + pdata_new.a[k].1 * dt / 2.0
        );
    }  
}


#[cfg(test)]
mod tests {
    use super::*;

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
        let dt = 0.1;
        let mut pdata = ParticleData::new(2, 2);
        pdata.x[0] = (10.0, 10.0);
        pdata.x[1] = (10.5, 10.0);
        let p1 = Particle { position: (10.0, 10.0), mass: 1.0, ..Default::default() };
        let p2 = Particle { position: (10.5, 10.0), mass: 1.0, ..Default::default() };
        let pg = PixelGrid::new(1000, 1000);
        let mut index = ParticleIndex::new(&pg); 
        let mut particles = vec![p1, p2];
        index.update(&pg, &particles);
        let mut prev_err = 99999.0;

        let pc = ParticleConstants {
            rho0_vec: vec![1.0, 0.7],
            c2_vec: vec![3.4, 3.8],
            mu_mat: vec![
                vec![2.5, 0.01], 
                vec![0.01, 3.0]
            ],
            s_mat: vec![
                vec![1.0, 0.0], 
                vec![0.0, 20.0]
            ],
            body_force: (0.0, -0.9)
        };

        let mut pdata_new = pdata.clone();
        assert!(pdata_new.density == pdata.density);

        for pi in 0..pdata.n_particles {
            _debug_print(&particles[pi]);
            _debug_print_ecs(pdata.get_particle_ref(pi));   
            _debug_print_ecs(pdata_new.get_particle_ref(pi));   
        }
        println!("");

        leapfrog_cal_forces(
            &pg, &mut index,
            &mut particles, 2,
            &pc, h
        );
        leapfrog_update_acceleration(&mut particles, 2);


        leapfrog_cal_forces_ecs(
            &pg, &mut index,
            &pdata,
            &mut pdata_new,
            &pc, h
        );
        leapfrog_update_acceleration_ecs(&mut pdata_new);

        println!("INIT LOOP");

        for pi in 0..pdata.n_particles {
            println!("{}", pi);
            _debug_print(&particles[pi]);
            _debug_print_ecs(pdata.get_particle_ref(pi));   
            _debug_print_ecs(pdata_new.get_particle_ref(pi));   
            assert!(particles[pi].acceleration == pdata_new.a[pi]);     
        }
        mem::swap(&mut pdata, &mut pdata_new);

        for _ in 0..20 {

            leapfrog(
                &pg, &mut index,
                &mut particles, 2,
                &pc, dt, h
            ); 
            
            leapfrog_ecs(
                &pg, &mut index,
                &pdata, &mut pdata_new,
                &pc, dt, h
            );  
            
            let mut new_err = 0.0;
            for p in &particles {
                let rho0 = pc.rho0_vec[p.particle_type];
                new_err += (p.density - rho0).abs()
            }    
            assert!(new_err <= prev_err);

            let mut new_err_ecs = 0.0;
            for pi in 0..pdata_new.n_fluid_particles {
                let rho0 = pc.rho0_vec[pdata_new.particle_type[pi]];
                new_err += (pdata_new.density[pi] - rho0).abs()
            }    
            assert!(new_err_ecs <= prev_err);

            prev_err = new_err; 

            for pi in 0..pdata_new.n_particles {
                assert!(particles[pi].position == pdata_new.x[pi]);     
            }
            mem::swap(&mut pdata, &mut pdata_new);

        }
        println!("");
        for pi in 0..pdata.n_particles {
            _debug_print(&particles[pi]);
            _debug_print_ecs(pdata.get_particle_ref(pi));   
        }
    }

}
