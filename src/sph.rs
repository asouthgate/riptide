use crate::particle::{Particle};
use crate::particle_ecs::*;
use crate::pixelgrid::PixelGrid;
use crate::particle_index::*;
use crate::kernels::*;


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
    println!("body: {:?} drag: {:?} hydro: {:?} surface: {:?}", 
        p.f_body, p.f_drag, p.f_hydro, p.f_surface
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
    pdata: &mut ParticleData, 
    pindex: &ParticleIndex,
    h: f32
) {
    for i in 0..pdata.n_particles {
        assert!(!pdata.x[i].0.is_nan());
        assert!(!pdata.x[i].1.is_nan());
        pdata.density[i] = 0.0;
        for j in pindex.neighbors[i].iter() {
            let rij = cal_dist(pdata.x[i], pdata.x[*j]);
            let contrib = cal_rho_ij(pdata.mass[*j], rij, h);
            pdata.density[i] += contrib; 
            assert!(!pdata.density[i].is_nan());
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
    pdata: &mut ParticleData, 
    pindex: &ParticleIndex,
    h: f32
) {
    for i in 0..pdata.n_fluid_particles { // ignore static particles
        let x = pdata.x[i].0;
        let y = pdata.x[i].1;
        assert!(!x.is_nan());
        assert!(!y.is_nan());
        pdata.f_pressure[i] = (0.0, 0.0);
        let mut ftot = (0.0, 0.0);
        for _nbrj in pindex.neighbors[i].iter() {
            let nbrj = *_nbrj;
            if nbrj == i {
                continue;
            }
            let dx = pdata.x[i].0 - pdata.x[nbrj].0;
            let dy = pdata.x[i].1 - pdata.x[nbrj].1;
            assert!(!dx.is_nan());
            assert!(!dy.is_nan());
            assert!(dx.powi(2) + dy.powi(2) > 0.0);
            let grad = debrun_spiky_kernel_grad(dx, dy, h);
            assert!(!grad.0.is_nan());
            assert!(!grad.1.is_nan());
            assert!(pdata.density[i] != 0.0);
            assert!(pdata.density[nbrj] != 0.0);
            let fij = cal_pressure_force_ij(
                pdata.pressure[i],
                pdata.pressure[nbrj],
                pdata.density[i],
                pdata.density[nbrj],
                pdata.mass[nbrj],
                grad
            );
            ftot.0 += fij.0;
            ftot.1 += fij.1;
            assert!(!fij.0.is_nan());
            assert!(!fij.1.is_nan());
        }
        pdata.f_pressure[i] = ftot;
        assert!(!pdata.f_pressure[i].0.is_nan());
        assert!(!pdata.f_pressure[i].1.is_nan());
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
    pdata: &mut ParticleData, 
    rho0_vec: &Vec<f32>, 
    c2_vec: &Vec<f32>
) {
    for k in 0..pdata.n_particles {
        let pk = pdata.particle_type[k];
        pdata.pressure[k] = cal_pressure(pdata.density[k], rho0_vec[pk], c2_vec[pk]);
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
    pdata: &mut ParticleData, 
    body_force: (f32, f32)
) {
    for k in 0..pdata.n_fluid_particles {
        pdata.f_body[k] = (body_force.0 * pdata.density[k], body_force.1 * pdata.density[k]);
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
    pdata: &mut ParticleData, 
    pindex: &ParticleIndex,
    h: f32, 
    s_mat: &Vec<Vec<f32>>
) {
    let c_s = (3.0 * PI) / (2.0 * h);
    for i in 0..pdata.n_fluid_particles {
        let x = pdata.x[i].0;
        let y = pdata.x[i].1;
        pdata.f_surface[i] = (0.0, 0.0);
        assert!(!x.is_nan());
        assert!(!y.is_nan());
        let mut f_surface_tot = (0.0, 0.0);
        for _nbrj in pindex.neighbors[i].iter() {
            let nbrj = *_nbrj;
            if nbrj == i {
                continue;
            }
            let dx = pdata.x[i].0 - pdata.x[nbrj].0;
            let dy = pdata.x[i].1 - pdata.x[nbrj].1;
            let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
            if r < h {
                let s = s_mat[pdata.particle_type[i]][pdata.particle_type[nbrj]];
                f_surface_tot.0 += s * (c_s * r).cos() * dx / r;
                f_surface_tot.1 += s * (c_s * r).cos() * dy / r;
            }
        }
        pdata.f_surface[i] = f_surface_tot;
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
    pdata: &mut ParticleData, 
    pindex: &ParticleIndex,
    h: f32, mu_mat: &Vec<Vec<f32>>
) {
    for i in 0..pdata.n_fluid_particles {
        let x = pdata.x[i].0;
        let y = pdata.x[i].1;
        pdata.f_viscous[i] = (0.0, 0.0);
        assert!(!x.is_nan());
        assert!(!y.is_nan());
        let mut f_viscous_tot = (0.0, 0.0);
        for _nbrj in pindex.neighbors[i].iter() {
            let nbrj = *_nbrj;
            if nbrj == i {
                continue;
            }
            let dx = pdata.x[i].0 - pdata.x[nbrj].0;
            let dy = pdata.x[i].1 - pdata.x[nbrj].1;
            let r2 = dx.powi(2) + dy.powi(2);
            let du = pdata.v[i].0 - pdata.v[nbrj].0;
            let dv = pdata.v[i].1 - pdata.v[nbrj].1;
            let grad = debrun_spiky_kernel_grad(dx, dy, h);
            let muij = mu_mat[pdata.particle_type[i]][pdata.particle_type[nbrj]];
            let a = 4.0 * pdata.mass[nbrj] / (pdata.density[nbrj] * pdata.density[i]);
            let b = (grad.0 * du) + (grad.1 * dv);
            let c = (dx / r2, dy / r2);
            f_viscous_tot.0 += muij * a * b * c.0;
            f_viscous_tot.1 += muij * a * b * c.1;

        }
        pdata.f_viscous[i] = f_viscous_tot;
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
    pdata: &mut ParticleData,
    particle_constants: &ParticleConstants,
    h: f32
) {
    let max_dist = h * 1.0;
    pindex.update_ecs(pg, pdata);
    pindex.update_neighbors(pg, pdata, max_dist);

    // update forces
    update_densities_ecs(pdata, pindex, h);
    update_body_forces_ecs(pdata, particle_constants.body_force);
    update_surface_forces_ecs(pdata, pindex, h, &particle_constants.s_mat);
    update_viscous_forces_ecs(pdata, pindex, h, &particle_constants.mu_mat);
    update_pressures_ecs(pdata, &particle_constants.rho0_vec, &particle_constants.c2_vec);
    update_pressure_forces_ecs(pdata, pindex, h);
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
    pdata: &mut ParticleData,
    particle_constants: &ParticleConstants, dt: f32,
    h: f32
) {

    for k in 0..pdata.n_fluid_particles {
        pdata.v[k] = (
            pdata.v[k].0 + pdata.a[k].0 * dt / 2.0,
            pdata.v[k].1 + pdata.a[k].1 * dt / 2.0
        );
        pdata.x[k] = (
            pdata.x[k].0 + dt * pdata.v[k].0,
            pdata.x[k].1 + dt * pdata.v[k].1
        );
    }  
    leapfrog_cal_forces_ecs(
        pg, index, pdata, particle_constants, h
    );
    leapfrog_update_acceleration_ecs(pdata);
    for k in 0..pdata.n_fluid_particles {
        pdata.v[k] = (
            pdata.v[k].0 + pdata.a[k].0 * dt / 2.0,
            pdata.v[k].1 + pdata.a[k].1 * dt / 2.0
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

        leapfrog_cal_forces(
            &pg, &mut index,
            &mut particles, 2,
            &pc, h
        );
        leapfrog_update_acceleration(&mut particles, 2);

        for _ in 0..20 {
            for p in &particles {
                _debug_print(p);
            }

            leapfrog(
                &pg, &mut index,
                &mut particles, 2,
                &pc, dt, h
            );  
            
            let mut new_err = 0.0;
            for p in &particles {
                let rho0 = pc.rho0_vec[p.particle_type];
                new_err += (p.density - rho0).abs()
            }    
            assert!(new_err <= prev_err);
            prev_err = new_err; 
        }
    }

}
