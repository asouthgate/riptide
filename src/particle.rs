use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;

pub struct Particle {
    pub position: (f32, f32),
    pub velocity: (f32, f32),
    pub acceleration: (f32, f32),
    pub f_hydro: (f32, f32),
    pub f_drag: (f32, f32),
    pub mass: f32,
    pub cdrag: f32,
    pub trail: Vec<(f32, f32)>,
    pub trail_length: usize,
    pub t: f32,
    pub lifespan: f32,
    pub rgba: (u8, u8, u8, u8)
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
            f_hydro: (0.0, 0.0),
            f_drag: (0.0, 0.0),
            mass: 1.0,
            cdrag: 1.0,
            trail: vec![],
            trail_length: 0,
            t: 0.0,
            lifespan: 0.0,
            rgba: (255, 255, 255, 255)
        }
    }
}

impl Particle {
    pub fn new(x: f32, y: f32) -> Self {
        Particle {
            position: (x, y), mass: 1.0, ..Default::default()
        }
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
}

pub fn length(u: f32, v: f32) -> f32 {
    (u.powf(2.0) + v.powf(2.0)).powf(0.5)
}

#[derive(Default)]
pub struct RigidBody {
    virtual_particles: Vec<Particle>,
    central_particle: Particle,
    relative_positions: Vec<(f32, f32)>,
}

impl RigidBody {
    pub fn new(x0: f32, y0: f32, mass_density: f32, positions: Vec<(f32, f32)>) -> Self {
        let mut vps = vec![];
        let n_vps = positions.len();
        for (x, y) in &positions {
            vps.push(Particle {
                position: (x0 + *x, y0 + *y), mass: mass_density,
                cdrag: 1.0 / n_vps as f32,
                ..Default::default()
            });
        }
        let n_particles = positions.len();
        RigidBody {
            virtual_particles: vps,
            central_particle: Particle { position: (x0, y0), mass: mass_density, ..Default::default() },
            relative_positions: positions,
        }
    }

    pub fn update_derivatives(&mut self, fs: &FluidState, pg: &PixelGrid, dt: f32) {
        let mut c = 0;
        self.central_particle.f_hydro = (0.0, 0.0);
        self.central_particle.f_drag = (0.0, 0.0);
        for p in &mut self.virtual_particles {
            // println!(
            //     "\t??? {} {} -> {} {}", p.velocity.0, p.velocity.1, p.f_drag.0, p.f_drag.1
            // );
            update_fluid_forces(fs, pg, p, dt);
            self.central_particle.f_hydro.0 += p.f_hydro.0;
            self.central_particle.f_hydro.1 += p.f_hydro.1;
            self.central_particle.f_drag.0 += p.f_drag.0;
            self.central_particle.f_drag.1 += p.f_drag.1;
            // println!(
            //     "\t({:.5} {:.5}) | ph ({:.5} {:.5}) pd ({:.5} {:.5}) a ({:.5} {:.5}) v ({:.5} {:.5}) |v| {:.5}",
            //     p.position.0, p.position.1,
            //     p.f_hydro.0, p.f_hydro.1,
            //     p.f_drag.0, p.f_drag.1,
            //     p.acceleration.0, p.acceleration.1,
            //     p.velocity.0, p.velocity.1,
            //     length(p.velocity.0, p.velocity.1)
            // );
            c += 1;
        }
        self.central_particle.f_hydro.0 /= self.virtual_particles.len() as f32;
        self.central_particle.f_hydro.1 /= self.virtual_particles.len() as f32;
        self.central_particle.f_drag.0 /= self.virtual_particles.len() as f32;
        self.central_particle.f_drag.1 /= self.virtual_particles.len() as f32;

        update_particle_derivatives(fs, pg, &mut self.central_particle, dt);

        // println!(
        //     "({:.5} {:.5}) | ph ({:.5} {:.5}) pd ({:.5} {:.5}) a ({:.5} {:.5}) v ({:.5} {:.5})",
        //     self.central_particle.position.0, self.central_particle.position.1,
        //     self.central_particle.f_hydro.0, self.central_particle.f_hydro.1,
        //     self.central_particle.f_drag.0, self.central_particle.f_drag.1,
        //     self.central_particle.acceleration.0, self.central_particle.acceleration.1,
        //     self.central_particle.velocity.0, self.central_particle.velocity.1,
        // );

        for p in &mut self.virtual_particles {
            p.acceleration.0 = self.central_particle.acceleration.0;
            p.acceleration.1 = self.central_particle.acceleration.1;
            p.velocity.0 = self.central_particle.velocity.0;
            p.velocity.1 = self.central_particle.velocity.1;
            // println!(
            //     "\t({:.5} {:.5}) | ({:.5} {:.5}) ({:.5} {:.5}) ({:.5} {:.5})",
            //     p.position.0, p.position.1,
            //     p.f_hydro.0, p.f_hydro.1,
            //     p.acceleration.0, p.acceleration.1,
            //     p.velocity.0, p.velocity.1,
            // );
        }
    }

    pub fn update_child_positions(&mut self) {
        let mut c = 0;
        for position in &self.relative_positions {
            self.virtual_particles[c].position.0 = self.central_particle.position.0 + position.0;
            self.virtual_particles[c].position.1 = self.central_particle.position.1 + position.1;
            c += 1;
        }
    }

    pub fn update_position(&mut self, fs: &FluidState, pg: &PixelGrid, dt: f32) {
        // now, we need to update all positions (all in the same direction)
        // if any one fails, they all fail to move
        let (mut dx, mut dy) = (1.0, 1.0); // start off as some non-zero value
        let mut c = 0;
        for particle in &mut self.virtual_particles {
            let (pdx, pdy) = cal_proposed_step(fs, pg, particle, dt);
            if dx != 0.0 {
                dx = pdx;
            }
            if dy != 0.0 {
                dy = pdy;
            }
            c += 1;
        }
        self.central_particle.position.0 += dx;
        self.central_particle.position.1 += dy;
        self.update_child_positions();
    }

    pub fn evolve(&mut self, fs: &FluidState, pg: &PixelGrid, dt: f32) {
        self.update_derivatives(fs, pg, dt);
        self.update_position(fs, pg, dt);
    }

    pub fn get_x(&mut self) -> &mut f32 {
        &mut self.central_particle.position.0
    }

    pub fn get_y(&mut self) -> &mut f32 {
        &mut self.central_particle.position.1
    }

    pub fn get_u(&mut self) -> &mut f32 {
        &mut self.central_particle.velocity.0
    }

    pub fn get_v(&mut self) -> &mut f32 {
        &mut self.central_particle.velocity.1
    }

}

fn update_fluid_forces(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, dt: f32) {
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

fn update_particle_derivatives(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, dt: f32) {
    p.acceleration.0 = p.f_hydro.0 / p.mass; // makes sense, the higher the mass the lower the acceleration
    p.acceleration.1 = p.f_hydro.1 / p.mass;
    p.acceleration.0 += p.f_drag.0 / p.mass;
    p.acceleration.1 += p.f_drag.1 / p.mass;
    p.velocity.0 += p.acceleration.0 * dt;
    p.velocity.1 += p.acceleration.1 * dt;
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

fn cal_proposed_step( 
    fs: &FluidState,
    pg: &PixelGrid,
    p: &mut Particle,
    dt: f32
) -> (f32, f32) {
    let prop_dx = p.velocity.0 * dt;
    let prop_dy = p.velocity.1 * dt;
    let mut ret = (0.0, 0.0);

    let ray = cal_ray(prop_dx, prop_dy);

    let mut collided_x = false;
    let mut collided_y = false;

    for (pdx, pdy) in ray {
        if !collided_x {
            match pg.sample_world(&(fs.boundary), p.get_x() + pdx, p.get_y()) {
                b if b > 0.0 => {
                    ret.0 = pdx;
                }
                _ => { collided_x = true; }
            }
        }
        if !collided_y {
            match pg.sample_world(&(fs.boundary), p.get_x(), p.get_y() + pdy) {
                b if b > 0.0 => {
                    ret.1 = pdy;
                }
                _ => { collided_y = true; }
            }
        }
    }
    ret
}

/// Update particle position.
///
/// # Returns
/// 
/// tuple of bools indicating whether move failed in x or y directions.
fn update_particle_position(
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

fn evolve_particle(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, dt: f32) {
    update_fluid_forces(fs, pg, p, dt);
    update_particle_derivatives(fs, pg, p, dt);
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
    fn test_evolve_rigidbody() {
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
        // first test a single particle, it should be the same as Particle
        let mut pos1 = vec![];
        pos1.push((0.0, 0.0));
        let mut rb1 = RigidBody::new(1.0, 1.0, 1.0, pos1);
        let mut p = Particle {
            position: (1.0, 1.0),
            mass: 1.0,
            lifespan: 4.0,
            ..Default::default()
        };
        for _it in 0..6 {
            evolve_particle(&fs, &pg, &mut p, 1.0);
            rb1.evolve(&fs, &pg, 1.0);
        }
        assert!(p.position.0 == 7.0); // started at 1, 1, did 6 steps
        assert!(p.position.1 == 4.0);
        assert!(*rb1.get_x() == p.position.0);
        assert!(*rb1.get_y() == p.position.1);

        let mut pos2 = vec![];
        pos2.push((0.0, 0.0));
        pos2.push((0.0, 1.0));
        let mut rb2 = RigidBody::new(1.0, 1.0, 1.0, pos2);
        pg.print_data(&fs.boundary);
        for _it in 0..10 {
            rb2.evolve(&fs, &pg, 1.0);
        }

        assert!(rb2.virtual_particles[0].get_y() == rb2.virtual_particles[1].get_y() - 1.0);
        assert!(rb2.virtual_particles[0].get_y() == *rb2.get_y());
        assert!(rb2.virtual_particles[0].get_x() == *rb2.get_x());

        assert!(rb2.virtual_particles[0].get_x() < 8.0);
        assert!(rb2.virtual_particles[0].get_y() < 7.0);
        assert!(rb2.virtual_particles[1].get_x() < 8.0);
        assert!(rb2.virtual_particles[1].get_y() < 8.0);
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
