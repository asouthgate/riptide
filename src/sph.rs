use crate::particle::{Particle, update_particle_velocity, update_particle_position};
use crate::fluid_state::FluidState;
use crate::pixelgrid::PixelGrid;

const PI: f32 = 3.141592653589793;

fn cubic_spline_fac(h: f32) -> f32 {
    10.0 / (7.0 * PI * h.powi(2))
}

fn cubic_spline_kernel(r: f32, h: f32) -> f32 {
    let norm = cubic_spline_fac(h);
    let q = r / h;
    let fq = match q {
        _q if _q > 2.0 => 0.0,
        _q if _q > 1.0 => 0.25 * (2.0 - q).powi(3),
        _ => 1.0 - 1.5 * q.powi(2) * (1.0 - 0.5 * q)
    };
    return norm * fq;
}

fn cubic_spline_kernel_dwdq(r: f32, h: f32) -> f32 {
    let norm = cubic_spline_fac(h);
    let q = r / h;
    let fq = match q {
        _q if _q > 2.0 => 0.0,
        _q if _q > 1.0 => -0.75 * (2.0 - q).powi(2),
        _ => - 3.0 * q * (1.0 - 0.75 * q)
    };
    return norm * fq;
}

fn cubic_spline_grad(dx: f32, dy: f32, r: f32, h: f32) -> (f32, f32) {
    match r {
        _r if _r > 0.0000001 => {
            let dwdq = cubic_spline_kernel_dwdq(r, h);
            let rhinv = 1.0 / (r * h);
            (
                rhinv * dx * dwdq,
                rhinv * dy * dwdq,
            )        
        }
        _ => (0.0, 0.0)
    }
}

fn cal_p(rho: f32, rho0: f32, c0: f32) -> f32 {
    rho0 + c0.powi(2) * (rho0 - rho)
}

// cal contribution of j to rho of i
fn cal_rho_ij(mj: f32, rij: f32) -> f32 {
    mj * cubic_spline_kernel(rij, 1.0)
}

fn cal_fviscosity_ij(
    dx: f32, dy: f32, r: f32,
    du: f32, dv: f32,
    rhoij: f32, // average density
    cij: f32, // avg speed of sound: speed of sound can vary by type
    hij: f32, // characteristic length between the two particles (can change)
    eta: f32, // fudge factor???
    alpha: f32, beta: f32, // computational constants, not specific to particles
) -> f32 {

    let vx = dx * du + dy * dv;

    let mut term = 0.0;
    if vx < 0.0 { // only if this product is less than zero do we have viscous drag
        let muij = (hij * vx)/(r.powi(2) + eta);
        term = -alpha * cij * muij + beta * muij * muij;
        term /= rhoij;
    }
    term    
}

fn cal_fpressure_ij(pi: f32, pj: f32, rhoi: f32, rhoj: f32) -> f32 {
    (pi / rhoi.powi(2)) + (pj / rhoj.powi(2))
}

/// calculate acceleration upon i from j
fn cal_acceleration_ij(
    dx: f32, dy: f32,
    du: f32, dv: f32,
    pi: f32, pj: f32,
    mj: f32,
    rhoi: f32, rhoj: f32,
    cij: f32,
    hij: f32,
    eta: f32,
    alpha: f32,
    beta: f32
) -> (f32, f32) {
    let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
    let rhoij = (rhoi + rhoj) / 2.0;
    let pterm = cal_fpressure_ij(pi, pj, rhoi, rhoj);
    let vterm = cal_fviscosity_ij(dx, dy, r, du, dv, rhoij, cij, hij, eta, alpha, beta);
    let gradk = cubic_spline_grad(dx, dy, r, hij);
    let total_term = -mj * pterm + vterm;
    (total_term * gradk.0, total_term * gradk.1)
}

fn cal_rho_i(i: usize, neighbors: &mut Vec<Particle>) {
    let n_particles = neighbors.len();
    neighbors[i].density = 0.0;
    for j in 0..n_particles {
        let dx = neighbors[i].get_x() - neighbors[j].get_x();
        let dy = neighbors[i].get_y() - neighbors[j].get_y();
        let rij = (dx.powi(2) + dy.powi(2)).powf(0.5);    
        neighbors[i].density += cal_rho_ij(neighbors[j].mass, rij);
    }
}

fn cal_p_i(particle: &mut Particle, rho0: f32, c0: f32) {
    particle.pressure = cal_p(particle.density, rho0, c0);
}

fn cal_acceleration_i(i: usize, neighbors: &mut Vec<Particle>, alpha: f32, beta: f32) {
    // compute fresh acceleration
    let n_particles = neighbors.len();
    neighbors[i].acceleration = (0.0, 0.0);
    for j in 0..n_particles {    
        let dx = neighbors[i].get_x() - neighbors[j].get_x();
        let dy = neighbors[i].get_y() - neighbors[j].get_y();
        let du = neighbors[i].get_u() - neighbors[j].get_u();
        let dv = neighbors[i].get_v() - neighbors[j].get_v();
        let cij = 1.0;
        let hij = 1.0;
        let eta = 0.0;
        let a = cal_acceleration_ij(
            dx, dy, du, dv,
            neighbors[i].pressure, neighbors[j].pressure,
            neighbors[j].mass,
            neighbors[i].density, neighbors[j].density,
            cij, hij, eta, alpha, beta
        );
        neighbors[i].acceleration.0 += a.0;
        neighbors[i].acceleration.1 += a.1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_kernel() {
        assert!(cubic_spline_kernel(0.00, 1.0) == 1.0 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel(0.50, 1.0) == 0.71875 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel(1.00, 1.0) == 0.25 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel(1.50, 1.0) == 0.03125 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel(2.00, 1.0) == 0.0);
        assert!(cubic_spline_kernel(2.01, 1.0) == 0.0);
        assert!(cubic_spline_kernel(10.0, 1.0) == 0.0);
    }

    #[test]
    fn test_cubic_spline_kernel_dwdq() {
        assert!(cubic_spline_kernel_dwdq(0.00, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(0.50, 1.0) == -0.9375 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(1.00, 1.0) == -0.75 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(1.50, 1.0) == -0.1875 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(2.00, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(2.01, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(10.0, 1.0) == 0.0);
    }

    #[test]
    fn test_kernel_lap() {

    }

    #[test]
    fn test_fviscosity_ij_trivial() {
        let mut fvij = cal_fviscosity_ij(
            0.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, // average density
            1.0, // avg speed of sound: speed of sound can vary by type
            1.0, // characteristic length between the two particles (can change)
            0.0, // fudge factor???
            1.0, 1.0, // computational constants, not specific to particles
        );
        assert!(fvij == 0.0);
        let mut dx: f32 = 1.0;
        let mut dy: f32 = 1.0;
        let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
        let mut fvij = cal_fviscosity_ij(
            dx, dy, r,
            -0.5, -0.5,
            1.0, // average density
            1.0, // avg speed of sound: speed of sound can vary by type
            1.0, // characteristic length between the two particles (can change)
            0.0, // fudge factor???
            1.0, 0.0, // computational constants, not specific to particles
        );
        assert!(fvij - 0.5 < 0.000001);
    }

    #[test]
    fn cal_acceleration_ij_trivial() {
        // both contributions will be zero
        let mut a = cal_acceleration_ij(
            10.0, 0.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0,
            1.0, 1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        );
        assert!(a == (0.0, 0.0));

        let mut a = cal_acceleration_ij(
            -1.0, 0.0, // it's minus one because we assume j to the right, xi - xj = 0 - 1 = -1
            0.0, 0.0,
            0.0, 1.0,
            1.0,
            1.0, 1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        );
        assert!(a.0 < 0.0);
        assert!(a.1 == 0.0);

        let mut a = cal_acceleration_ij(
            0.0, 1.0, // it's plus one because we assume j to the below, xi - xj = 0 - - 1 = 1
            0.0, 0.0,
            0.0, 1.0,
            1.0,
            1.0, 1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        );
        assert!(a.0 == 0.0);
        assert!(a.1 > 0.0);
    }

    #[test]
    fn test_cal_acceleration_i_justpressure_symmetry() {
        // This test places 8 particles around a central particle, one at a time
        // and makes sure the forces they exert balance when the pressure and distance
        // is symmetric
        let mut p1 = Particle{
            position: (0.0, 0.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        let mut p2 = Particle{
            position: (1.0, 0.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        let mut p3 = Particle{
            position: (-1.0, 0.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        let mut nbrs: Vec<Particle> = vec![p2];
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0);
        assert!(p1.acceleration.0 < 0.0);
        assert!(p1.acceleration.1 == 0.0);

        nbrs.push(p3);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0);
        assert!(p1.acceleration.0 == 0.0);
        assert!(p1.acceleration.1 == 0.0);

        let mut p4 = Particle{
            position: (-1.0, -1.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        nbrs.push(p4);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0);
        assert!(p1.acceleration.0 >= 0.0);
        assert!(p1.acceleration.1 >= 0.0);

        let mut p5 = Particle{
            position: (-1.0, 1.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        nbrs.push(p5);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0);
        assert!(p1.acceleration.0 >= 0.0);
        assert!(p1.acceleration.1 == 0.0);

        let mut p6 = Particle{
            position: (1.0, -1.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        nbrs.push(p6);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0);
        assert!(p1.acceleration.0 >= 0.0);
        assert!(p1.acceleration.1 >= 0.0);

        let mut p7 = Particle{
            position: (1.0, 1.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        nbrs.push(p7);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0);
        assert!(p1.acceleration.0 == 0.0);
        assert!(p1.acceleration.1 == 0.0);
    }

    #[test]
    fn test_two_relax_in_box() {
        let pg = PixelGrid::new(3, 6);
        let mut fs = FluidState::new(&pg);
        pg.print_data(&fs.boundary);
        let rho0 = 1.0;
        let c0 = 1.0;
        let dt = 1.0;
        let mut p1 = Particle{
            position: (2.0, 1.0), mass: 1.0,
            .. Default::default()
        };
        let mut p2 = Particle{
            position: (3.0, 1.0), mass: 1.0,
            .. Default::default()
        };
        let mut particles: Vec<Particle> = vec![p1, p2];
        for _it in 0..5 {
            for k in 0..particles.len() {
                cal_rho_i(k, &mut particles);
            }
            for k in 0..particles.len() {
                cal_p_i(&mut particles[k], rho0, c0);
            }
            for k in 0..particles.len() {
                cal_acceleration_i(k, &mut particles, 0.0, 0.0);
            }
            for k in 0..particles.len() {
                print!("{} ", k);
                particles[k].print();
                update_particle_velocity(&mut particles[k], dt);
                update_particle_position(&fs, &pg, &mut particles[k], dt);
                print!("{} ", k);
                particles[k].print();
                println!("-----------------")
            }
            println!("");
        }
    }
}