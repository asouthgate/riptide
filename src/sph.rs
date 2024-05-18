use crate::particle::{Particle, update_particle_position, attenuate_particle_velocity_at_boundary};
use crate::fluid_state::FluidState;
use crate::pixelgrid::PixelGrid;
use crate::particle_index::*;
use crate::kernels::*;
use std::time::{Instant};

const PI: f32 = 3.141592653589793;

pub struct ParticleConstants {
    /// These structures are intentionally simple
    /// We do not want high abstraction level
    /// We need to be able to port to GPU code easily
    pub rho0_vec: Vec<f32>, // Vector; one for each type of Particle
    pub c2_vec: Vec<f32>, // speed of sound squared: one for each type of Particle
    pub mu_mat: Vec<Vec<f32>>, // viscosity; one for each PAIRWISE COMBINATION of Particle types
    pub s_mat: Vec<Vec<f32>>, // viscosity; one for each PAIRWISE COMBINATION of Particle types
}

fn debug_print(p: &Particle) {
    println!("body: {:?} drag: {:?} hydro: {:?} surface: {:?}", 
        p.f_body, p.f_drag, p.f_hydro, p.f_surface
    );
}

fn cal_pressure(rho: f32, rho0: f32, c: f32) -> f32 {
    c * (rho - rho0)
}

fn cal_rho_ij(mj: f32, rij: f32, h: f32) -> f32 {
    mj * debrun_spiky_kernel(rij, h)
}

fn cal_pressure_force_coefficient(pi: f32, pj: f32, rhoi: f32, rhoj: f32, mj: f32) -> f32 {
    ( (pi/rhoi.powi(2)) + (pj/rhoj.powi(2)) ) * mj
}

// NOTE WELL, THERE IS A FORMAL, REDUNDANT MULTIPLICATION BY Pi
// In optimization, this could be removed.
// It is needed for a = f / rho to make plain sense. 
// ommitting this pi factor, this would be the acceleration, not the force
fn cal_pressure_force_ij(pi: f32, pj: f32, rhoi: f32, rhoj: f32, mj: f32, grad: (f32, f32)) -> (f32, f32) {
    let pforce_coefficient = - pi * cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, mj);
    (pforce_coefficient * grad.0, pforce_coefficient * grad.1)
}

// fn cal_pressure_acceleration_ij(pi: f32, force: (f32, f32)) -> (f32, f32) {
//     (force.0 / pi, force.1 / pi)
// }

pub fn update_densities(
    particles: &mut Vec<Particle>, h: f32
) {
    for i in 0..particles.len() {
        let x = particles[i].get_x();
        let y = particles[i].get_y();
        assert!(!x.is_nan());
        assert!(!y.is_nan());
        // let mut nbrs = index.get_nbrs(&pg, x, y, max_dist);
        // nbrs.push(i);
        particles[i].density = 0.0;
        let mut density_sum = 0.0;
        // println!("{}", nbrs.len());
        for nbrj in particles[i].nbrs.iter() {
            let dx = x - particles[*nbrj].get_x();
            let dy = y - particles[*nbrj].get_y();
            let rij = (dx.powi(2) + dy.powi(2)).powf(0.5);
            let contrib = cal_rho_ij(particles[*nbrj].mass, rij, h);
            density_sum += contrib; 
            assert!(!density_sum.is_nan());
        }
        particles[i].density = density_sum;
        // correct the density for isolated particles, otherwise it can be too high
        // let wmax = cal_rho_ij(particles[i].mass, 0.0, h);
        // let rhocorr = (particles[i].density + (n_nbrs as f32 - 1.0) * wmax) / wmax;
        // println!("rho {}, wmax {}, nnbrs {}, corrected_density: {}", particles[i].density, wmax, n_nbrs, rhocorr);
        // particles[i].density = rhocorr;
    }
}

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
            // println!("{} {} | {} {} {}", particles[i].get_x(), particles[i].get_y(), dx, dy, dx.powi(2) + dy.powi(2));
            assert!(dx.powi(2) + dy.powi(2) > 0.0);
            let grad = debrun_spiky_kernel_grad(dx, dy, h);
            // println!("\t {} {} {} {} {} {}", i, nbrj, dx, dy, grad.0, grad.1);
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
            assert!(!fij.0.is_nan());
            assert!(!fij.1.is_nan());

            ftot.0 += fij.0;
            ftot.1 += fij.1;

            assert!(!particles[i].f_hydro.0.is_nan());
            assert!(!particles[i].f_hydro.1.is_nan());
        }
        particles[i].f_hydro = ftot;
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
                // println!("{} {} {} {} {}", 
                //     s, particles[i].particle_type, particles[nbrj].particle_type, f_surface_tot.0, f_surface_tot.1
                // )
            }
        }
        particles[i].f_surface = f_surface_tot;
    }
}

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
            let r = (r2).powf(0.5);
            // calculate grad
            // let grad = debrun_spiky_kernel_grad(dx, dy, h);
            // calculate velocity
            let du = particles[i].get_u() - particles[nbrj].get_u();
            let dv = particles[i].get_v() - particles[nbrj].get_v();
            // take dot product
            // let dot = du * grad.0 + dv * grad.1;
            // let lap = debrun_spiky_kernel_lap(r, h);
            let grad = debrun_spiky_kernel_grad(dx, dy, h);

            // take min (needs to be negative)
            // multiply by mu
            let muij = mu_mat[particles[i].particle_type][particles[nbrj].particle_type];
            let a = 4.0 * particles[nbrj].mass / (particles[nbrj].density * particles[i].density);
            let b = (grad.0 * du) + (grad.1 * dv);
            let c = (dx / r2, dy / r2);

            f_drag_tot.0 += muij *  a * b * c.0;
            f_drag_tot.1 += muij *  a * b * c.1;

        }
        particles[i].f_drag = f_drag_tot;
    }
}


pub fn update_viscous_forces_and_velocities(
    particles: &mut Vec<Particle>, 
    n_real_particles: usize, h: f32, mu_mat: &Vec<Vec<f32>>, dt: f32
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
            let r = (r2).powf(0.5);
            // calculate grad
            // let grad = debrun_spiky_kernel_grad(dx, dy, h);
            // calculate velocity
            let du = particles[i].get_u() - particles[nbrj].get_u();
            let dv = particles[i].get_v() - particles[nbrj].get_v();
            // take dot product
            // let dot = du * grad.0 + dv * grad.1;
            // let lap = debrun_spiky_kernel_lap(r, h);
            let grad = debrun_spiky_kernel_grad(dx, dy, h);

            // take min (needs to be negative)
            // multiply by mu
            let muij = mu_mat[particles[i].particle_type][particles[nbrj].particle_type];
            let a = 4.0 * particles[nbrj].mass / (particles[nbrj].density * particles[i].density);
            let b = (grad.0 * du) + (grad.1 * dv);
            let c = (dx / r2, dy / r2);

            f_drag_tot.0 += muij *  a * b * c.0;
            f_drag_tot.1 += muij *  a * b * c.1;

        }
        particles[i].f_drag = f_drag_tot;
        // NB: THERE IS A REASON THIS IS NOT IN A SEPARATE FUCNTION
        // FORCE/VEL ARE ITERATED
        particles[i].velocity = (
            particles[i].velocity.0 + (dt / particles[i].density) * (particles[i].f_body.0 + particles[i].f_drag.0 + particles[i].f_surface.0),
            particles[i].velocity.1 + (dt / particles[i].density) * (particles[i].f_body.1 + particles[i].f_drag.1 + particles[i].f_surface.1)
        );
    }
}

pub fn update_velocities(particles: &mut Vec<Particle>, n_real_particles: usize, dt: f32) {
    for k in 0..n_real_particles {
        particles[k].velocity = (
            particles[k].velocity.0 + (dt / particles[k].density) * particles[k].f_body.0,
            particles[k].velocity.1 + (dt / particles[k].density) * particles[k].f_body.1
        );
    }
}


pub fn update_velocities_and_positions(pg: &PixelGrid, fs: &FluidState, particles: &mut Vec<Particle>, n_real_particles: usize, dt: f32) {
    for k in 0..n_real_particles {
        particles[k].velocity = (
            particles[k].velocity.0 + (dt / particles[k].density) * (particles[k].f_hydro.0),
            particles[k].velocity.1 + (dt / particles[k].density) * (particles[k].f_hydro.1)
        );
        // attenuate_particle_velocity_at_boundary(pg, fs, &mut particles[k], 0.1);
        // particles[k].position = (
        //     particles[k].position.0 + dt * particles[k].velocity.0,
        //     particles[k].position.1 + dt * particles[k].velocity.1
        // );
        // update_particle_position(fs, pg, &mut particles[k], dt)
        particles[k].position = (
            particles[k].position.0 + dt * particles[k].velocity.0,
            particles[k].position.1 + dt * particles[k].velocity.1
        );
    }
}

pub fn leapfrog_update_acceleration(particles: &mut Vec<Particle>, n_real_particles: usize, dt: f32) {
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

pub fn leapfrog_cal_forces(
    pg: &PixelGrid, fs: &FluidState, index: &mut ParticleIndex,
    particles: &mut Vec<Particle>, n_real_particles: usize,
    particle_constants: &ParticleConstants, dt: f32,
    h: f32, body_force: (f32, f32)
) {
    let max_dist = h * 1.0;
    index.update(pg, particles);
    let mut avg_nbrs = 0.0;
    for i in 0..particles.len() {
        particles[i].nbrs = index.get_nbrs(&pg, particles[i].get_x(), particles[i].get_y(), max_dist);
        cull_nbrs(i, particles, h);
        avg_nbrs += particles[i].nbrs.len() as f32;
    }
    avg_nbrs /= particles.len() as f32;

    // update forces
    update_densities(particles, h);
    update_body_forces(particles, n_real_particles, body_force);
    update_surface_forces(particles, n_real_particles, h, &particle_constants.s_mat);
    update_viscous_forces(particles, n_real_particles, h, &particle_constants.mu_mat);
    update_pressures(particles, &particle_constants.rho0_vec, &particle_constants.c2_vec);
    update_pressure_forces(particles, h, n_real_particles);
}

pub fn leapfrog(
    pg: &PixelGrid, fs: &FluidState, index: &mut ParticleIndex,
    particles: &mut Vec<Particle>, n_real_particles: usize,
    particle_constants: &ParticleConstants, dt: f32,
    h: f32, body_force: (f32, f32)
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
    leapfrog_cal_forces(pg, fs, index, particles, n_real_particles, particle_constants, dt, h, body_force);
    leapfrog_update_acceleration(particles, n_real_particles, dt);
    for k in 0..n_real_particles {
        particles[k].velocity = (
            particles[k].velocity.0 + particles[k].acceleration.0 * dt / 2.0,
            particles[k].velocity.1 + particles[k].acceleration.1 * dt / 2.0
        );
    }  
}

pub fn integrate(
    pg: &PixelGrid, fs: &FluidState, index: &mut ParticleIndex,
    particles: &mut Vec<Particle>, n_real_particles: usize,
    particle_constants: &ParticleConstants, dt: f32,
    h: f32, body_force: (f32, f32)
) {
    let t0 = Instant::now();
    let max_dist = h * 1.0;
    index.update(pg, particles);
    let mut avg_nbrs = 0.0;
    for i in 0..particles.len() {
        particles[i].nbrs = index.get_nbrs(&pg, particles[i].get_x(), particles[i].get_y(), max_dist);
        cull_nbrs(i, particles, h);
        // for nbrj in &particles[i].nbrs {
        //     println!("{} {}", h, particles[*nbrj].dist(&particles[i]));
        // }
        avg_nbrs += particles[i].nbrs.len() as f32;
    }
    avg_nbrs /= particles.len() as f32;
    println!("Average nbr size: {}", avg_nbrs);
    let tindex = Instant::now();
    update_densities(particles, h);
    let tdensity = Instant::now();
    // compute viscous and body forces & update velocity (one particle at a time)
    // when i say one at a time, update velocity i, which effects calc of velocity i + 1 
    // it's data dependent, not parallelizable updates
    update_body_forces(particles, n_real_particles, body_force);
    // update_velocities(particles, n_real_particles, dt);
    update_surface_forces(
        particles, n_real_particles, h, &particle_constants.s_mat,
    );
    update_viscous_forces_and_velocities(
        particles, n_real_particles, h, &particle_constants.mu_mat, dt
    );
    update_pressures(particles, &particle_constants.rho0_vec, &particle_constants.c2_vec);
    let tpressure = Instant::now();
    update_pressure_forces(particles, h, n_real_particles);
    let tpressureforce = Instant::now();
    update_velocities_and_positions(pg, fs, particles, n_real_particles, dt);    
    let tpos = Instant::now();
    println!(
        "{:?} {:?} {:?} {:?} {:?}",
        tindex.duration_since(t0),
        tdensity.duration_since(tindex),
        tpressure.duration_since(tdensity),
        tpressureforce.duration_since(tpressure),
        tpos.duration_since(tpressureforce),
    );
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
        println!("{} {}", cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, m), (2 + 1));
        assert!(cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, m) == (2.0 + 1.0));
    }

    #[test]
    fn test_2_particles() {
        let h: f32 = 2.0;
        let rho0_vec = vec![1.0];
        let c2_vec = vec![1.0];
        let dt = 0.1;
        let p1 = Particle { position: (10.0, 10.0), mass: 1.0, ..Default::default() };
        let p2 = Particle { position: (10.5, 10.0), mass: 1.0, ..Default::default() };
        let pg = PixelGrid::new(1000, 1000);
        let fs = FluidState::new(&pg);
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
        };

        leapfrog_cal_forces(
            &pg, &fs, &mut index,
            &mut particles, 2,
            &pc, dt, h, (0.0, -0.9)
        );
        leapfrog_update_acceleration(&mut particles, 2, dt);

        for _ in 0..20 {
            for p in &particles {
                debug_print(p);
            }

            leapfrog(
                &pg, &fs, &mut index,
                &mut particles, 2,
                &pc, dt, h, (0.0, -0.9)
            );  
            
            let mut new_err = 0.0;
            for p in &particles {
                let rho0 = rho0_vec[p.particle_type];
                new_err += (p.density - rho0).abs()
            }    
            assert!(new_err <= prev_err);
            prev_err = new_err; 
        }
    }

}
