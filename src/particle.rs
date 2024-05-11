use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;

pub struct Particle {
    pub position: (f32, f32),
    pub velocity: (f32, f32),
    pub acceleration: (f32, f32),
    pub pressure: f32,
    pub density: f32,
    pub h: f32, // characteristic length
    pub f_hydro: (f32, f32), // hydrostatic pressure force
    pub f_drag: (f32, f32), // viscous drag force
    pub f_body: (f32, f32),
    // TODO: this is a waste of memory
    pub viscosity: f32, // fluid viscosity
    pub eq_density: f32, // equilibrium density for comparing state eqn to pressure
    pub c_sound: f32, // again, for state equation
    pub vboost: (f32, f32),
    pub mass: f32,
    pub cdrag: f32,
    pub trail: Vec<(f32, f32)>,
    pub nbrs: Vec<usize>,
    pub trail_length: usize,
    pub t: f32,
    pub lifespan: f32,
    pub rgba: (u8, u8, u8, u8),
    pub particle_type: u32 // encoded as an int, define your own
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
            f_hydro: (0.0, 0.0),
            f_drag: (0.0, 0.0),
            f_body: (0.0, 0.0),
            viscosity: 0.0,
            eq_density: 0.0,
            c_sound: 1.0,
            pressure: 0.0,
            density: 1.0,
            vboost: (0.0, 0.0),
            mass: 1.0,
            cdrag: 1.0,
            h: 1.0,
            trail: vec![],
            nbrs: vec![],
            trail_length: 0,
            t: 0.0,
            lifespan: 0.0,
            rgba: (255, 255, 255, 255),
            particle_type: 0
        }
    }
}

impl Particle {
    pub fn new(x: f32, y: f32) -> Self {
        Particle {
            position: (x, y), mass: 1.0, ..Default::default()
        }
    }
    pub fn evolve(&mut self, fs: &FluidState, pg: &PixelGrid, dt: f32) {
        evolve_particle(fs, pg, self, dt)
    }
    pub fn is_dead(&self) -> bool{
        self.t > self.lifespan
    }
    pub fn get_x(&self) -> f32 {
        self.position.0
    }
    pub fn get_y(&self) -> f32 {
        self.position.1
    }
    pub fn get_u(&self) -> f32 {
        self.velocity.0
    }
    pub fn get_v(&self) -> f32 {
        self.velocity.1
    }
    pub fn print(&self) {
        println!(
            "({:.5} {:.5}) | ph ({:.5} {:.5}) pd ({:.5} {:.5}) a ({:.5} {:.5}) v ({:.5} {:.5})",
            self.position.0, self.position.1,
            self.f_hydro.0, self.f_hydro.1,
            self.f_drag.0, self.f_drag.1,
            self.acceleration.0, self.acceleration.1,
            self.velocity.0, self.velocity.1,
        );
    }
}

pub fn length(u: f32, v: f32) -> f32 {
    (u.powf(2.0) + v.powf(2.0)).powf(0.5)
}


pub fn update_fluid_forces(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, _dt: f32) { // dt could be used
    // fluid forces, for unit density and unit area, is just velocity
    // this can be calculated from the momentum flux
    // or, imagine water is little balls
    // a flow of z per second collisions is z newtons per second
    // and ofc you can relate mass to density * area 

    // however, what's also important is _relative_ velocity
    // since if our object is going the same velocity as the flow
    // there will be no momentum flux
    let fu = pg.sample_bilinear_world(&fs.u, p.get_x(), p.get_y());
    let fv = pg.sample_bilinear_world(&fs.v, p.get_x(), p.get_y());
    p.f_hydro = (fu, fv);
    // let cdrag = 1.0;
    // f_drag is 
    p.f_drag = (-p.get_u() * p.cdrag, -p.get_v() * p.cdrag);
}

pub fn update_particle_velocity(p: &mut Particle, dt: f32) {
    p.velocity.0 += p.acceleration.0 * dt;
    p.velocity.1 += p.acceleration.1 * dt;    
}

pub fn attenuate_particle_velocity_at_boundary(pg: &PixelGrid, fs: &FluidState, p: &mut Particle, factor: f32) {
    let bleft = pg.sample_world(&fs.boundary, p.get_x() - 1.0, p.get_y());
    let bright = pg.sample_world(&fs.boundary, p.get_x() + 1.0, p.get_y());
    let bup = pg.sample_world(&fs.boundary, p.get_x(), p.get_y() + 1.0);
    let bdown = pg.sample_world(&fs.boundary, p.get_x(), p.get_y() - 1.0);
    if bleft == 0.0 && p.velocity.0 < 0.0 {
        p.velocity.0 = -p.velocity.0 * factor;
    }
    if bright == 0.0 && p.velocity.0 > 0.0 {
        p.velocity.0 = -p.velocity.0 * factor;
    }
    if bdown == 0.0 && p.velocity.1 < 0.0{
        p.velocity.1 = -p.velocity.1 * factor;
    }
    if bup == 0.0 && p.velocity.1 > 0.0 {
        p.velocity.1 = -p.velocity.1 * factor;
    }
}

pub fn update_particle_derivatives(p: &mut Particle, dt: f32) {
    p.acceleration.0 = p.f_hydro.0 / p.mass; // makes sense, the higher the mass the lower the acceleration
    p.acceleration.1 = p.f_hydro.1 / p.mass;
    p.acceleration.0 += p.f_drag.0 / p.mass;
    p.acceleration.1 += p.f_drag.1 / p.mass;
    update_particle_velocity(p, dt);
}

fn cal_ray(x: f32, y: f32) -> Vec<(f32, f32)> {
    let sign_x = x.signum();
    let sign_y = y.signum();
    let x_abs = x.abs();
    let y_abs = y.abs();
    // now build a ray to (prop_dx, prop_dy)
    let mut ray: Vec<(f32, f32)> = vec![];
    let maxdrs = x_abs.max(y_abs).trunc() + 1.0;
    for drs in 0..maxdrs as i32 {
        ray.push((
            sign_x * (drs as f32).min(x_abs),
            sign_y * (drs as f32).min(y_abs)
        ))
    }
    ray.push((x, y));
    ray
}

pub fn cal_proposed_step( 
    fs: &FluidState,
    pg: &PixelGrid,
    p: &mut Particle,
    dt: f32
) -> (f32, f32) {
    let prop_dx = p.velocity.0 * dt;
    let prop_dy = p.velocity.1 * dt;
    let mut ret = (0.0, 0.0);

    let ray = cal_ray(prop_dx, prop_dy);

    let mut not_collided_x = true;
    let mut not_collided_y = true;
    
    for (pdx, pdy) in ray {
        // assert!(pg.sample_world(&(fs.boundary), p.get_x() + ret.0, p.get_y() + ret.1) > 0.0);
        not_collided_x = not_collided_x && (pg.sample_world(&(fs.boundary), p.get_x() + pdx, p.get_y() + ret.1) > 0.0);
        if not_collided_x {
            ret.0 = pdx;
        }
        not_collided_y = not_collided_y && (pg.sample_world(&(fs.boundary), p.get_x() + ret.0, p.get_y() + pdy) > 0.0);
        if not_collided_y {
            ret.1 = pdy;
        }
    }
    // assert!(pg.sample_world(&(fs.boundary), p.get_x() + ret.0, p.get_y() + ret.1) > 0.0);
    ret
}

/// Update particle position.
///
/// # Returns
/// 
/// tuple of bools indicating whether move failed in x or y directions.
pub fn update_particle_position(
    fs: &FluidState,
    pg: &PixelGrid,
    p: &mut Particle,
    dt: f32
) {
    p.t += dt;
    let (dx, dy) = cal_proposed_step(fs, pg, p, dt);
    p.position.0 += dx;
    p.position.1 += dy;
}

pub fn evolve_particle(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, dt: f32) {
    update_fluid_forces(fs, pg, p, dt);
    update_particle_derivatives(p, dt);
    update_particle_position(fs, pg, p, dt);
    if p.trail.len() > p.trail_length {
        let (lastx, lasty) = p.trail[p.trail.len()-1];
        if lastx.floor() != p.get_x().floor() || lasty.floor() != p.get_y().floor() {
            p.trail.push((p.get_x(), p.get_y()));
        }
    } else {
        p.trail.push((p.get_x(), p.get_y()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_particles_never_cross_boundary() {
        let pg = PixelGrid::new(20, 40);  
        let mut fs = FluidState::new(&pg);
        let seed_value = 42;
        let mut rng = StdRng::seed_from_u64(seed_value);
        for j in 0..pg.n {
            for i in 0..pg.m {
                let ak = i * pg.n + j;
                fs.u[ak] = rng.gen_range(0.0..5.0);
                fs.v[ak] = rng.gen_range(0.0..5.0);
            }
        }
        for _bi in 0..200 {
            let ak = rng.gen_range(0..pg.mn);
            fs.boundary[ak] = 0.0;
        }
        pg.print_data(&fs.u);
        pg.print_data(&fs.v);
        pg.print_data(&fs.boundary);
        let mut particles: Vec<Particle> = vec![];
        for j in 1..pg.n-1 {
            for i in 1..pg.m-1 {
                let ak = i * pg.n + j;
                if fs.boundary[ak] > 0.0 {
                    particles.push(Particle {
                        position: (j as f32, i as f32),
                        .. Default::default()
                    })
                }
            }
        }
        for _it in 0..100 {
            for p in &mut particles {
                let ak = pg.xy2ak(p.get_x(), p.get_y());
                p.evolve(&fs, &pg, 1.0);
                assert!(pg.sample(&fs.boundary, p.get_x(), p.get_y()) > 0.0);
            }
        }   
    }

    #[test]
    fn test_cal_proposed_step() {
        let pg = PixelGrid::new(10, 10);  
        let mut fs = FluidState::new(&pg);
        // for _ak in 0..pg.mn {
        //     fs.u[_ak] = 10.0;
        //     fs.v[_ak] = 0.0;
        // }
        for j in 0..pg.n {
            let ak = 6 * pg.n + j;
            fs.boundary[ak] = 0.0;
        }
        for i in 0..pg.m {
            let ak = i * pg.n + 8;
            fs.boundary[ak] = 0.0;
        }
        let mut p = Particle {
            position: (1.0, 1.0),
            velocity: (0.81, 0.52),
            mass: 1.0,
            lifespan: 4.0,
            ..Default::default()
        };
        pg.print_data(&fs.u);
        pg.print_data(&fs.v);
        pg.print_data(&fs.boundary);
        let (mut px, mut py) = (0.0, 0.0);
        for _it in 0..100 {
            (px, py) = cal_proposed_step(&fs, &pg, &mut p, 1.0);
            p.position.0 += px;
            p.position.1 += py;
        }
        assert!(p.get_x() <= 8.0);
        assert!(p.get_y() <= 6.0);
        assert!(p.get_x() >= 6.0);
        assert!(p.get_y() >= 4.0);
    }

    #[test]
    fn test_cal_ray() {
        let mut ray = cal_ray(5.1, 2.3);
        let raylen = ray.len();
        assert!(ray[raylen-1].0 == 5.1);
        assert!(ray[raylen-2].0 == 5.0);
        assert!(ray[raylen-3].0 == 4.0);
        assert!(ray[raylen-1].1 == 2.3);
        assert!(ray[raylen-2].1 == 2.3);
        assert!(ray[raylen-3].1 == 2.3);

        ray = cal_ray(2.3, -5.1);
        assert!(ray[raylen-1].0 == 2.3);
        assert!(ray[raylen-2].0 == 2.3);
        assert!(ray[raylen-1].1 == -5.1);
        assert!(ray[raylen-2].1 == -5.0);

        ray = cal_ray(-2.3, -5.1);
        assert!(ray[raylen-1].0 == -2.3);
        assert!(ray[raylen-2].0 == -2.3);
        assert!(ray[raylen-1].1 == -5.1);
        assert!(ray[raylen-2].1 == -5.0);

        ray = cal_ray(-2.3, 5.1);
        assert!(ray[raylen-1].0 == -2.3);
        assert!(ray[raylen-2].0 == -2.3);
        assert!(ray[raylen-1].1 == 5.1);
        assert!(ray[raylen-2].1 == 5.0);
    }


    #[test]
    fn test_evolve_particle_mass_one() {
        // for a particle of unit mass, it should flow the same as the fluid
        // since the fluid is unit density
        let pg = PixelGrid::new(10, 10);  
        let mut fs = FluidState::new(&pg);
        for _ak in 0..pg.mn {
            fs.u[_ak] = 1.0;
            fs.v[_ak] = 0.5;
        }
        for j in 0..pg.n {
            let ak = 8 * pg.n + j;
            fs.boundary[ak] = 0.0;
        }
        for i in 0..pg.m {
            let ak = i * pg.n + 8;
            fs.boundary[ak] = 0.0;
        }
        let mut p = Particle {
            position: (1.0, 1.0),
            mass: 1.0,
            lifespan: 4.0,
            ..Default::default()
        };
        pg.print_data(&fs.u);
        pg.print_data(&fs.v);
        pg.print_data(&fs.boundary);

        println!("init: {} {} <- {} {} <- {} {}", p.position.0, p.position.1, p.get_u(), p.get_v(), p.acceleration.0, p.acceleration.1);
        evolve_particle(&fs, &pg, &mut p, 1.0);
        println!("{} {} <- {} {} <- {} {}", p.position.0, p.position.1, p.get_u(), p.get_v(), p.acceleration.0, p.acceleration.1);
        assert!(! p.is_dead());
        for _it in 0..5 {
            evolve_particle(&fs, &pg, &mut p, 1.0);
            println!("{} {} <- {} {} <- {} {}", p.position.0, p.position.1, p.get_u(), p.get_v(), p.acceleration.0, p.acceleration.1);
        }
        assert!(p.is_dead());
        assert!(p.position.0 == 7.0); // started at 1, 1, did 6 steps
        assert!(p.position.1 == 4.0);

        // now test the boundary
        for _it in 0..50 {
            evolve_particle(&fs, &pg, &mut p, 1.0);
            println!("{} {}", p.position.0, p.position.1);
        }
        assert!(p.position.0 <= 8.0);
        assert!(p.position.1 <= 8.0);
    }

}
