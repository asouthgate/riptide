use crate::particle::{Particle, update_particle_velocity, update_particle_position, attenuate_particle_velocity_at_boundary};
use crate::fluid_state::FluidState;
use crate::pixelgrid::PixelGrid;
use std::time::{Instant, Duration};

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
        _r if _r > 1e-13 => {
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
fn cal_rho_ij(mj: f32, rij: f32, h: f32) -> f32 {
    mj * cubic_spline_kernel(rij, h)
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
    // println!("\t dx: {} dy: {} r: {} pterm: {} vterm: {} grad: ({} {})", dx, dy, r, pterm, vterm, gradk.0, gradk.1);

    let total_term = -mj * pterm + vterm;
    (total_term * gradk.0, total_term * gradk.1)
}

fn cal_rho_i(i: usize, neighbors: &mut Vec<Particle>, h: f32) {
    let n_particles = neighbors.len();
    neighbors[i].density = 0.0;
    for j in 0..n_particles {
        let dx = neighbors[i].get_x() - neighbors[j].get_x();
        let dy = neighbors[i].get_y() - neighbors[j].get_y();
        let rij = (dx.powi(2) + dy.powi(2)).powf(0.5);
        neighbors[i].density += cal_rho_ij(neighbors[j].mass, rij, h);
    }
}

fn cal_acceleration_i_nbrs(i: usize, nbr_inds: & Vec<usize>, particles: &mut Vec<Particle>, alpha: f32, beta: f32, hij: f32) {
    // compute fresh acceleration
    let n_particles = particles.len();
    particles[i].acceleration = (0.0, 0.0);
    for _j in nbr_inds {
        let j = *_j;
        let dx = particles[i].get_x() - particles[j].get_x();
        let dy = particles[i].get_y() - particles[j].get_y();
        let du = particles[i].get_u() - particles[j].get_u();
        let dv = particles[i].get_v() - particles[j].get_v();
        let cij = 1.0;
        let eta = 0.0;
        let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
        let a = cal_acceleration_ij(
            dx, dy, du, dv,
            particles[i].pressure, particles[j].pressure,
            particles[j].mass,
            particles[i].density, particles[j].density,
            cij, hij, eta, alpha, beta
        );
        particles[i].acceleration.0 += a.0;
        particles[i].acceleration.1 += a.1;
        let i2 = particles[j].get_y().trunc() as usize;
        let j2 = particles[j].get_x().trunc() as usize;
        // println!("{} {} {} -> {} {}", i2, j2, r, a.0, a.1);
        // println!("");
    }
}
fn cal_rho_i_nbrs(i: usize, nbr_inds: &Vec<usize>, particles: &mut Vec<Particle>, h: f32) {
    let n_particles = particles.len();
    particles[i].density = cal_rho_ij(particles[i].mass, 0.0, h);
    for j in nbr_inds {
        let dx = particles[i].get_x() - particles[*j].get_x();
        let dy = particles[i].get_y() - particles[*j].get_y();
        let rij = (dx.powi(2) + dy.powi(2)).powf(0.5);
        particles[i].density += cal_rho_ij(particles[*j].mass, rij, h);
    }
}

fn cal_p_i(particle: &mut Particle, rho0: f32, c0: f32) {
    particle.pressure = cal_p(particle.density, rho0, c0);
}

fn cal_acceleration_i(i: usize, neighbors: &mut Vec<Particle>, alpha: f32, beta: f32, hij: f32) {
    // compute fresh acceleration
    let n_particles = neighbors.len();
    neighbors[i].acceleration = (0.0, 0.0);
    for j in 0..n_particles {    
        let dx = neighbors[i].get_x() - neighbors[j].get_x();
        let dy = neighbors[i].get_y() - neighbors[j].get_y();
        let du = neighbors[i].get_u() - neighbors[j].get_u();
        let dv = neighbors[i].get_v() - neighbors[j].get_v();
        let cij = 1.0;
        let eta = 0.0;
        let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
        let a = cal_acceleration_ij(
            dx, dy, du, dv,
            neighbors[i].pressure, neighbors[j].pressure,
            neighbors[j].mass,
            neighbors[i].density, neighbors[j].density,
            cij, hij, eta, alpha, beta
        );
        neighbors[i].acceleration.0 += a.0;
        neighbors[i].acceleration.1 += a.1;
        let i2 = neighbors[j].get_y().trunc() as usize;
        let j2 = neighbors[j].get_x().trunc() as usize;
        // println!("{} {} {} -> {} {}", i2, j2, r, a.0, a.1);
        // println!("");
    }
}

pub fn get_ghost_particles_naive(fs: &FluidState, pg: &PixelGrid) -> Vec<Particle> {
    let mut res = vec![];
    for i in 0..pg.m {
        for j in 0..pg.n {
            let ak = i * pg.n + j;
            if fs.boundary[ak] == 0.0 {
                res.push(Particle{
                    position: (j as f32 + pg.dx / 2.0, i as f32 + pg.dy / 2.0),
                    mass: 1.0,
                    density: 1.0,
                    pressure: 0.0,
                    .. Default::default()
                })
            }
        }
    }
    res
}

pub fn get_ghost_box(fs: &mut FluidState, pg: &PixelGrid, i0: i32, ie: i32, j0: i32, je: i32, n: i32) -> Vec<Particle> {
    let mut res = vec![];
    for i in i0..ie {
        let (mut x, mut y) = pg.worldxy2xy(j0 as f32, i as f32);
        let mut ak = pg.xy2ak(x, y);
        fs.boundary[ak] = 0.0;
        let (mut x, mut y) = pg.worldxy2xy(je as f32, i as f32);
        let mut ak = pg.xy2ak(x, y);
        fs.boundary[ak] = 0.0;
        res.push(Particle{
            position: (j0 as f32 + 0.5, i as f32 + 0.5),
            mass: 1.0,
            density: 1.0,
            pressure: 0.0,
            .. Default::default()
        });
        res.push(Particle{
            position: (je as f32 + 0.5, i as f32 + 0.5),
            mass: 1.0,
            density: 1.0,
            pressure: 0.0,
            .. Default::default()
        });
    }
    for j in j0..je {
        let (mut x, mut y) = pg.worldxy2xy(j as f32, i0 as f32);
        let mut ak = pg.xy2ak(x, y);
        fs.boundary[ak] = 0.0;
        let (mut x, mut y) = pg.worldxy2xy(j as f32, ie as f32);
        let mut ak = pg.xy2ak(x, y);
        fs.boundary[ak] = 0.0;
        res.push(Particle{
            position: (j as f32 + 0.5, i0 as f32 + 0.5),
            mass: 1.0,
            density: 1.0,
            pressure: 0.0,
            .. Default::default()
        });
        res.push(Particle{
            position: (j as f32 + 0.5, ie as f32 + 0.5),
            mass: 1.0,
            density: 1.0,
            pressure: 0.0,
            .. Default::default()
        });   
    }
    res
}

pub struct ParticleIndex {
    slots: Vec<Vec<usize>>
}

impl ParticleIndex {
    pub fn new(pg: &PixelGrid) -> Self {
        let mut slots = vec![];
        for ak in 0..pg.mn {
            slots.push(vec![]);
        }
        ParticleIndex {
            slots: slots
        }
    }
    pub fn update(&mut self, pg: &PixelGrid, particles: &Vec<Particle>) {
        for ak in 0..pg.mn {
            self.slots[ak].clear();
        }
        for pi in 0..particles.len() {
            let (x, y) = pg.worldxy2xy(particles[pi].get_x(), particles[pi].get_y());
            let ak = pg.xy2ak(x, y);
            self.slots[ak].push(pi);
        }
    }
    pub fn get_nbrs(&self, pg: &PixelGrid, wx: f32, wy: f32, dist: f32) -> Vec<usize> {
        let mut result = vec![];
        let (x, y) = pg.worldxy2xy(wx, wy);
        let i0 = (y - dist).max(0.0) as usize;
        let ie = (y + dist + 1.0).min(pg.m as f32 + 1.0) as usize;
        let j0 = (x - dist).max(0.0) as usize;
        let je = (x + dist + 1.0).min(pg.n as f32 + 1.0) as usize;
        for y2 in i0..ie {
            for x2 in j0..je {
                let ak = pg.xy2ak(x2 as f32, y2 as f32);
                for ind in &self.slots[ak] {
                    result.push(*ind);
                }
                if result.len() as f32 > dist.powi(2) {
                    // println!("ALERT: better way to do this, step over area once first, like bottom of pyramid");
                    return result;
                }
            }
        }
        return result;
    }
}

pub fn update_all_particles(
    pg: &PixelGrid, fs: &FluidState, 
    particles: &mut Vec<Particle>, n_real_particles: usize,
    rho0: f32, c0: f32, h: f32, dt: f32, force: (f32, f32), alpha: f32, beta: f32
) {
    let mut index = ParticleIndex::new(pg);
    index.update(pg, particles);
    let dist = 2.0;
    println!("nreal: {}, n: {} n^2: {}", n_real_particles, particles.len(), particles.len().pow(2));
    let t0 = Instant::now();
    for k in 0..n_real_particles {
        let x = particles[k].get_x();
        let y = particles[k].get_y();
        assert!(!x.is_nan());
        assert!(!y.is_nan());
        // println!("???? {} {} {}", x, y, particles[k].density);
        let nbrs = index.get_nbrs(&pg, x, y, dist);
        // println!("!!({} {}) nbrs size: {}, density: {}", x, y, nbrs.len(), particles[k].density);
        cal_rho_i_nbrs(k, &nbrs, particles, h);
        // println!("???? {} {} {}", x, y, particles[k].density);
        // assert!(!particles[k].density.is_nan());
    }
    println!("");
    let trho = Instant::now();
    for k in 0..particles.len() {
        cal_p_i(&mut particles[k], rho0, c0);
        // println!("???? {}", particles[k].pressure);
        assert!(!particles[k].density.is_nan());

    }
    let tp = Instant::now();
    for k in 0..n_real_particles {
        let x = particles[k].get_x();
        let y = particles[k].get_y();
        let nbrs = index.get_nbrs(&pg, x, y, dist);
        // println!("??nbrs size: {}", nbrs.len());
        cal_acceleration_i_nbrs(k, &nbrs, particles, alpha, beta, h);
        particles[k].acceleration.0 += force.0;
        particles[k].acceleration.1 += force.1;
    }
    let ta = Instant::now();
    for k in 0..n_real_particles {
        attenuate_particle_velocity_at_boundary(&pg, &fs, &mut particles[k]);
        update_particle_velocity(&mut particles[k], dt);
        update_particle_position(&fs, &pg, &mut particles[k], dt);
    }
    let tpos = Instant::now();
    println!(
        "{:?} {:?} {:?} {:?}",
        trho.duration_since(t0),
        tp.duration_since(trho),
        ta.duration_since(tp),
        tpos.duration_since(ta),
    )

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
    fn test_cubic_spline_kernel_dwdq_unit_h() {
        assert!(cubic_spline_kernel_dwdq(0.00, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(0.50, 1.0) == -0.9375 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(1.00, 1.0) == -0.75 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(1.50, 1.0) == -0.1875 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(2.00, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(2.01, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(10.0, 1.0) == 0.0);
    }

    // #[test]
    // fn test_cubic_spline_kernel_dwdq_bigh() {
    //     let h = 1.0;
    //     assert!(cubic_spline_kernel_dwdq(0.00, h) == 0.0);
    //     let mut xs = vec![];
    //     let maxk = 100;
    //     for k in 0..maxk {
    //         xs.push(h * 2.0 * ( k as f32 / maxk as f32));
    //     }
    //     let mut ys = vec![];
    //     for x in xs {
    //         let y = cubic_spline_kernel_dwdq(x, h);
    //         println!("? {:.5} {:.5} ", x, y);
    //         ys.push(y);
    //     }
    //     println!("")
    //     // println!("{} {} {} {} {}", v0, v05, v1, v, v4, v5);
    // }

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
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0, 1.0);
        assert!(p1.acceleration.0 < 0.0);
        assert!(p1.acceleration.1 == 0.0);

        nbrs.push(p3);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0, 1.0);
        assert!(p1.acceleration.0 == 0.0);
        assert!(p1.acceleration.1 == 0.0);

        let mut p4 = Particle{
            position: (-1.0, -1.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        nbrs.push(p4);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0, 1.0);
        assert!(p1.acceleration.0 >= 0.0);
        assert!(p1.acceleration.1 >= 0.0);

        let mut p5 = Particle{
            position: (-1.0, 1.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        nbrs.push(p5);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0, 1.0);
        assert!(p1.acceleration.0 >= 0.0);
        assert!(p1.acceleration.1 == 0.0);

        let mut p6 = Particle{
            position: (1.0, -1.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        nbrs.push(p6);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0, 1.0);
        assert!(p1.acceleration.0 >= 0.0);
        assert!(p1.acceleration.1 >= 0.0);

        let mut p7 = Particle{
            position: (1.0, 1.0), velocity: (0.0, 0.0),
            mass: 1.0, density: 1.0, pressure: 1.0,
            .. Default::default()
        };
        nbrs.push(p7);
        cal_acceleration_i(0, &mut nbrs, 0.0, 0.0, 1.0);
        assert!(p1.acceleration.0 == 0.0);
        assert!(p1.acceleration.1 == 0.0);
    }

    #[test]
    fn test_quad_in_box() {
        // 4 pixels inside a box should have perfectly symmetrical pressure on them
        // assert that as we evolve, density and pressure of particles inside are all the same
        let pg = PixelGrid::new(4,4);
        let mut fs = FluidState::new(&pg);
        let rho0 = 1.0;
        let c0 = 0.01;
        let dt = 1.0;
        let h = 1.0;
        let halfm = pg.m as f32 / 2.0;

        let mut particles: Vec<Particle> = vec![];
        for i in 1..pg.m-1 {
            for j in 1..pg.n-1 {
                particles.push(Particle{
                    position: (j as f32 + pg.dx * 0.5, i as f32 + pg.dy * 0.5), mass: 1.0,
                    .. Default::default()
                });
                fs.divergence[i * pg.n + j] = 1.0;
            }
        }
        let n_real_particles = particles.len();
        pg.print_data(&fs.boundary);
        pg.print_data(&fs.divergence);

        particles.extend(get_ghost_particles_naive(&fs, &pg));

        for _it in 0..500 {
            for k in 0..particles.len() {
                cal_rho_i(k, &mut particles, h);
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                println!("density: {} {} {}", i, j, particles[k].density);
            }
            for k in 0..n_real_particles {
                println!("density: {} {}", particles[k].density, particles[0].density);
                assert!( (particles[k].density-particles[0].density).abs() < 0.1 );
            }
            for k in 0..particles.len() {
                cal_p_i(&mut particles[k], rho0, c0);
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                println!("pressure: {} {} {}", i, j, particles[k].pressure);
            }
            for k in 0..n_real_particles {
                println!("pressure: {} {}", particles[k].pressure, particles[0].pressure);
                assert!( (particles[k].pressure-particles[0].pressure).abs() < 0.1 );
            }
            for k in 0..n_real_particles {
                cal_acceleration_i(k, &mut particles, 0.0, 0.0, h);
            }
            for k in 0..n_real_particles {
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                let ak = (i * pg.n) + j;
                fs.divergence[ak] += 1.0;
            }

            for k in 0..n_real_particles {
                let i = particles[k].get_y().round() as usize;
                let j = particles[k].get_x().round() as usize;

                attenuate_particle_velocity_at_boundary(&pg, &fs, &mut particles[k]);
                update_particle_velocity(&mut particles[k], dt);
                print!("{} {} ", i ,j);
                particles[k].print();
                update_particle_position(&fs, &pg, &mut particles[k], dt);
                print!("{} {} ", i ,j);
                particles[k].print();
                println!("-----------------");
            }
            for k in n_real_particles..particles.len() {
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                let ak = (i * pg.n) + j;
                fs.divergence[ak] = 2.0;
            }
            pg.print_data(&fs.divergence);
            println!("");
        }
    }

    #[test]
    fn test_pipe_flow() {
        // 4 pixels inside a box should have perfectly symmetrical pressure on them
        // assert that as we evolve, density and pressure of particles inside are all the same
        let pg = PixelGrid::new(20,4);
        let mut fs = FluidState::new(&pg);
        let rho0 = 1.0;
        let c0 = 0.1;
        let dt = 1.0;
        let h = 1.0;
        let halfm = pg.m as f32 / 2.0;

        let mut particles: Vec<Particle> = vec![];
        for i in 1..pg.m-1 {
            for j in 1..pg.n-1 {
                particles.push(Particle{
                    position: (j as f32 + pg.dx * 0.5, i as f32 + pg.dy * 0.5), mass: 1.0,
                    .. Default::default()
                });
                fs.divergence[i * pg.n + j] = 1.0;
            }
        }
        particles[0].velocity.1 = 1.0;
        particles[1].velocity.1 = 1.0;
        let n_real_particles = particles.len();
        pg.print_data(&fs.boundary);
        pg.print_data(&fs.divergence);

        particles.extend(get_ghost_particles_naive(&fs, &pg));

        for _it in 0..4 {
            for k in n_real_particles..particles.len() {
                cal_rho_i(k, &mut particles, h);
            }
            for k in n_real_particles..particles.len() {
                cal_p_i(&mut particles[k], rho0, c0);
            }
            for k in 0..particles.len() {
                cal_rho_i(k, &mut particles, h);
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                // println!("density: {} {} {}", i, j, particles[k].density);
            }
            for k in 0..particles.len() {
                cal_p_i(&mut particles[k], rho0, c0);
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                // println!("pressure: {} {} {}", i, j, particles[k].pressure);
            }
            for k in 0..n_real_particles {
                cal_acceleration_i(k, &mut particles, 0.0, 0.0, h);
            }
            

            for k in 0..n_real_particles {
                let i = particles[k].get_y().round() as usize;
                let j = particles[k].get_x().round() as usize;

                attenuate_particle_velocity_at_boundary(&pg, &fs, &mut particles[k]);
                update_particle_velocity(&mut particles[k], dt);
                print!("{} {} ", i ,j);
                particles[k].print();
                update_particle_position(&fs, &pg, &mut particles[k], dt);
                print!("{} {} ", i ,j);
                particles[k].print();
                println!("-----------------");
            }
            for ak in 0..pg.mn {
                fs.divergence[ak] = 0.0;
            }
            for k in 0..n_real_particles {
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                let ak = (i * pg.n) + j;
                fs.divergence[ak] += 1.0;
            }
            for k in n_real_particles..particles.len() {
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                let ak = (i * pg.n) + j;
                fs.divergence[ak] = 2.0;
            }
            pg.print_data(&fs.divergence);
            println!("");
            for k in 0..n_real_particles {
                let i = particles[k].get_y().trunc() as usize;
                let j = particles[k].get_x().trunc() as usize;
                let ak = i * pg.n + j;
                fs.laplacian[ak] += particles[k].velocity.1 / n_real_particles as f32;
            }
            pg.print_data(&fs.laplacian);    
        }
        println!("");
    }

    #[test]
    fn test_particle_index_count() {
        let pg = PixelGrid::new(10, 10);
        let mut particles = vec![];
        for i in 0..pg.m {
            for j in 0..pg.n {
                particles.push(Particle{
                    position: (j as f32, i as f32),
                    .. Default::default()
                });
            }
        }
        let mut index = ParticleIndex::new(&pg);
        index.update(&pg, &particles);
        println!("");
        for i in 0..pg.m {
            for j in 0..pg.n {
                print!("{} {}: ", i, j);
                let nbrs = index.get_nbrs(&pg, i as f32, j as f32, 1.0);
                assert!(nbrs.len() >= 4);
                assert!(nbrs.len() <= 9);
                for pu in nbrs {
                    print!("{} ", pu);
                }
                println!("");
            }
        }
    }

    #[test]
    fn test_particle_retrieval() {
        // in this scenario, only two particles; only a few slots have them
        let pg = PixelGrid::new(10, 10);
        let mut particles = vec![];
        particles.push(Particle{
            position: (5.5, 5.5),
            .. Default::default()
        });
        particles.push(Particle{
            position: (6.5, 6.5),
            .. Default::default()
        });
        
        let mut index = ParticleIndex::new(&pg);
        index.update(&pg, &particles);
        println!("");
        for i in 0..pg.m {
            for j in 0..2 {
                let nbrs = index.get_nbrs(&pg, i as f32, j as f32, 1.0);
                println!("{} {}: {} ", i, j, nbrs.len());
                assert!(nbrs.len() == 0);
            }
        }
        println!("");
        for i in 0..pg.m {
            for j in 8..10 {
                let nbrs = index.get_nbrs(&pg, i as f32, j as f32, 1.0);
                println!("{} {}: {} ", i, j, nbrs.len());
                assert!(nbrs.len() == 0);
            }
        }
        println!("");
        for i in 5..7 {
            for j in 5..7 {
                let nbrs = index.get_nbrs(&pg, i as f32, j as f32, 1.0);
                assert!(nbrs.len() == 2);
                println!("{} {}: {} ", i, j, nbrs.len());
            }
        }
    }

}