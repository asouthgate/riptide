use crate::particle::{Particle, update_particle_velocity, update_particle_position, attenuate_particle_velocity_at_boundary};
use crate::fluid_state::FluidState;
use crate::pixelgrid::PixelGrid;
use crate::particle_index::*;
use crate::kernels::*;
use std::time::{Instant, Duration};


fn cal_pressure(rho: f32, rho0: f32, c: f32) -> f32 {
    c * (rho0 - rho)
}

fn cal_rho_ij(mj: f32, rij: f32, h: f32) -> f32 {
    mj * cubic_spline_kernel(rij, h)
}

fn cal_pressure_force_coefficient(pi: f32, pj: f32, rhoi: f32, rhoj: f32, mj: f32) -> f32 {
    ( (pi/rhoi.powi(2)) + (pj/rhoj.powi(2)) ) * mj
}

// NOTE WELL, THERE IS A FORMAL, REDUNDANT MULTIPLICATION BY Pi
// In optimization, this could be removed.
// It is needed for a = f / rho to make plain sense. 
// ommitting this pi factor, this would be the acceleration, not the force
fn cal_pressure_force_ij(pi: f32, pj: f32, rhoi: f32, rhoj: f32, mj: f32, grad: (f32, f32)) -> (f32, f32) {
    let pforce_coefficient = pi * cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, mj);
    (pforce_coefficient * grad.0, pforce_coefficient * grad.1)
}

fn cal_pressure_acceleration_ij(pi: f32, force: (f32, f32)) -> (f32, f32) {
    (force.0 / pi, force.1 / pi)
}

// fn cal_divergence_ij( 
//     dx: f32, dy: f32, r: f32,
//     du: f32, dv: f32,
//     mj: f32,
//     rhoij: f32, // average density
//     hij: f32, // characteristic length between the two particles (can change)
// ) -> (f32, f32) {
//     let w = cubic_spline_grad(dx, dy, r, hij);
//     (du * w.0, dv * w.1)
// }

// fn cal_fviscosity_ij(
//     dx: f32, dy: f32, r: f32,
//     du: f32, dv: f32,
//     rhoij: f32, // average density
//     cij: f32, // avg speed of sound: speed of sound can vary by type
//     hij: f32, // characteristic length between the two particles (can change)
//     eta: f32, // fudge factor???
//     alpha: f32, beta: f32, // computational constants, not specific to particles
// ) -> f32 {

//     let vx = dx * du + dy * dv;

//     let mut term = 0.0;
//     if vx < 0.0 { // only if this product is less than zero do we have viscous drag
//         let muij = (hij * vx)/(r.powi(2) + eta);
//         term = -alpha * cij * muij + beta * muij * muij;
//         term /= rhoij;
//     }
//     term    
// }

// fn cal_fpressure_ij(pi: f32, pj: f32, rhoi: f32, rhoj: f32) -> f32 {
//     (pi / rhoi.powi(2)) + (pj / rhoj.powi(2))
// }

// /// calculate acceleration upon i from j
// fn cal_acceleration_ij(
//     dx: f32, dy: f32,
//     du: f32, dv: f32,
//     pi: f32, pj: f32,
//     mj: f32,
//     rhoi: f32, rhoj: f32,
//     hij: f32,
//     muij: f32
// ) -> (f32, f32) {
//     let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
//     let rhoij = (rhoi + rhoj) / 2.0;
//     let pterm = cal_fpressure_ij(pi, pj, rhoi, rhoj);
//     // let pterm = (pi-pj) / rhoij;
//     // let mut vterm = cal_divergence_ij(dx, dy, r, du, dv, mj, rhoij, hij);
//     // vterm.0 *= muij / rhoij;
//     // vterm.1 *= muij / rhoij;
//     let dwdr = debrun_spiky_kernel_dwdr(r, hij);
//     assert!(r != 0.0);
//     let drdx = dx / r;
//     let drdy = dy / r;
//     // println!("\t dx: {} dy: {} r: {} pterm: {} vterm: {} {} grad: ({} {})", dx, dy, r, pterm, vterm.0, vterm.1, gradk.0, gradk.1);
//     // (-mj * (pterm + vterm.0) * gradk.0, -mj * (pterm + vterm.1) * gradk.1)
//     (-mj * pterm * dwdr * drdx, -mj * pterm * dwdr * drdy)
// }

// fn cal_rho_i(i: usize, neighbors: &mut Vec<Particle>, h: f32) {
//     let n_particles = neighbors.len();
//     neighbors[i].density = 0.0;
//     for j in 0..n_particles {
//         let dx = neighbors[i].get_x() - neighbors[j].get_x();
//         let dy = neighbors[i].get_y() - neighbors[j].get_y();
//         let rij = (dx.powi(2) + dy.powi(2)).powf(0.5);
//         neighbors[i].density += cal_rho_ij(neighbors[j].mass, rij, h);
//     }
// }

// fn cal_acceleration_i_nbrs(
//     i: usize, nbr_inds: & Vec<usize>, particles: &mut Vec<Particle>, 
//     muij: f32, hij: f32
// ) {
//     // compute fresh acceleration
//     let n_particles = particles.len();
//     particles[i].acceleration = (0.0, 0.0);
//     for _j in nbr_inds {
//         let j = *_j;
//         if j == i {
//            continue; 
//         }
//         let dx = particles[i].get_x() - particles[j].get_x();
//         let dy = particles[i].get_y() - particles[j].get_y();
//         let du = particles[i].get_u() - particles[j].get_u();
//         let dv = particles[i].get_v() - particles[j].get_v();
//         let cij = 1.0;
//         let eta = 0.0;
//         let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
//         let a = cal_acceleration_ij(
//             dx, dy, du, dv,
//             particles[i].pressure, particles[j].pressure,
//             particles[j].mass,
//             particles[i].density, particles[j].density,
//             hij, muij
//         );
//         particles[i].acceleration.0 += a.0;
//         particles[i].acceleration.1 += a.1;
//         let i2 = particles[j].get_y().trunc() as usize;
//         let j2 = particles[j].get_x().trunc() as usize;
//     }
// }

// fn cal_rho_i_nbrs(i: usize, nbr_inds: &Vec<usize>, particles: &mut Vec<Particle>, h: f32) {
//     let n_particles = particles.len();
//     particles[i].density = cal_rho_ij(particles[i].mass, 0.0, h);
//     for j in nbr_inds {
//         let dx = particles[i].get_x() - particles[*j].get_x();
//         let dy = particles[i].get_y() - particles[*j].get_y();
//         let rij = (dx.powi(2) + dy.powi(2)).powf(0.5);
//         particles[i].density += cal_rho_ij(particles[*j].mass, rij, h);
//     }
// }

// fn cal_p_i(particle: &mut Particle, rho0: f32, c0: f32) {
//     particle.pressure = cal_p(particle.density, rho0, c0);
// }



// pub fn leapfrog(
//     pg: &PixelGrid, fs: &FluidState, index: &mut ParticleIndex,
//     particles: &mut Vec<Particle>, n_real_particles: usize,
//     rho0: f32, c0: f32, h: f32, dt: f32, force: (f32, f32), muij: f32
// ) {
//     index.update(pg, particles);
//     let max_dist = 10.0;
//     let t0 = Instant::now();
//     for k in 0..particles.len() {
//         let x = particles[k].get_x();
//         let y = particles[k].get_y();
//         assert!(!x.is_nan());
//         assert!(!y.is_nan());
//         let nbrs = index.get_nbrs(&pg, x, y, max_dist);
//         cal_rho_i_nbrs(k, &nbrs, particles, h);

//         // in this scheme, we do x* = xt + dt vt-1/2
//         // we use p.velocity = vt-1/2
//         // then x* = x* + dt^2 F^k (x*) /m, where F^k is some force, e.g. pressure
//         // then xt+1 = x*
//         // then vt+1/2 = (xt+1 - x_t) / dt
//         // we only have one F^k so we have
//         let vtm12 = particle.velocity; // vt-1/2
//         let x_ = particle.position; // xt
//         let x_ = (x_.0 + dt * vtm12.0, x_.1 + dt * vtm12.1);
//         let pi = particle.pressure;
//         let rhoi = particle.density;
//         let pforce = (0.0, 0.0);
//         for nbrj in nbrs {
//             let pj = nbrs[j].pressure;
//             let rhoj = nbrs[j].density; 
//             let pforce_ij = cal_pressure_force_ij(dx, dy, pi, pj, rhoi, rhoj, mj);
//             pforce = (pforce.0 + pforce_c * dx / r, pforce.1 + pforce_c * dy / r);
//         }
        
//         let x_ += dt.powi(2) * F(x_) / m;
//         let vtp12 = (x_ - x) / dt;
//         let x = x_;


//     }
// }

// pub fn update_all_particles(
//     pg: &PixelGrid, fs: &FluidState, index: &mut ParticleIndex,
//     particles: &mut Vec<Particle>, n_real_particles: usize,
//     rho0: f32, c0: f32, h: f32, dt: f32, force: (f32, f32), muij: f32
// ) {
//     index.update(pg, particles);
//     let max_dist = 2.0;
//     let t0 = Instant::now();
//     for k in 0..particles.len() {
//         let x = particles[k].get_x();
//         let y = particles[k].get_y();
//         assert!(!x.is_nan());
//         assert!(!y.is_nan());
//         let nbrs = index.get_nbrs(&pg, x, y, max_dist);
//         cal_rho_i_nbrs(k, &nbrs, particles, h);
//     }
//     // println!("");
//     let trho = Instant::now();
//     for k in 0..particles.len() {
//         cal_p_i(&mut particles[k], rho0, c0);
//         // println!("???? {}", particles[k].pressure);
//         assert!(!particles[k].density.is_nan());

//     }
//     let tp = Instant::now();
//     for k in 0..n_real_particles {
//         let x = particles[k].get_x();
//         let y = particles[k].get_y();
//         let nbrs = index.get_nbrs(&pg, x, y, max_dist);
//         // println!("??nbrs size: {}", nbrs.len());
//         cal_acceleration_i_nbrs(k, &nbrs, particles, muij, h);
//         particles[k].acceleration.0 += force.0;
//         particles[k].acceleration.1 += force.1;
//     }
//     let ta = Instant::now();
//     for k in 0..n_real_particles {
//         attenuate_particle_velocity_at_boundary(&pg, &fs, &mut particles[k]);
//         update_particle_velocity(&mut particles[k], dt);
//         update_particle_position(&fs, &pg, &mut particles[k], dt);
//     }
//     let tpos = Instant::now();
//     // println!(
//     //     "{:?} {:?} {:?} {:?}",
//     //     trho.duration_since(t0),
//     //     tp.duration_since(trho),
//     //     ta.duration_since(tp),
//     //     tpos.duration_since(ta),
//     // )

// }

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_cal_p() {
        assert!(cal_pressure(1.0, 0.0, 1.0) == -1.0);
        assert!(cal_pressure(1.0, 0.0, 2.0) == -2.0);
    }

    #[test]
    fn test_cal_pressure_force_coeff_ij() {
        let rhoi = 1.0;
        let rhoj = 2.0;
        let pi = cal_pressure(rhoi, 0.0, 2.0); 
        let pj = cal_pressure(rhoj, 0.0, 2.0);
        let m = 1.0;
        assert!(pi == -2.0);
        assert!(pj == -4.0);
        println!("{} {}", cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, m), (-2 + -1));
        assert!(cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, m) == (-2.0 + -1.0));
    }

    #[test]
    fn test_cal_pressure_force_ij() {
        let h: f32 = 1.329;
        let dx: f32 = 0.1361;
        let dy: f32 = 0.9981;
        let r: f32 = cal_r(dx, dy);
        let h = 1.8;
        let grad = debrun_spiky_kernel_grad(dx, dy, h);
        let rhoi = 1.0;
        let rhoj = 2.0;
        let pi = cal_pressure(rhoi, 0.0, 1.0);
        let pj = cal_pressure(rhoj, 0.0, 1.0);
        assert!(pj < pi);
        let mj = 1.0;
        let pf = cal_pressure_force_ij(pi, pj, rhoi, rhoj, mj, grad);
    }
}
