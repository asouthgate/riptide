use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;

#[derive(Default)]
pub struct Particle {
    pub position: (f32, f32),
    pub velocity: (f32, f32),
    pub acceleration: (f32, f32),
    pub mass: f32,
    pub trail: Vec<(f32, f32)>,
    pub trail_length: usize,
    pub t: f32,
    pub lifespan: f32,
    pub rgba: (u8, u8, u8, u8)
}

impl Particle {
    pub fn new(x: f32, y: f32) -> Self {
        Particle {
            position: (x, y), velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0), mass: 1.0,
            trail: vec![], trail_length: 0, t: 0.0,
            lifespan: 0.0, rgba: (0, 0, 0, 0)
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

struct RigidBody {
    virtual_particles: Vec<Particle>,
}

impl RigidBody {
    pub fn new(positions: Vec<(f32, f32)>) -> Self {
        let mut vps = vec![];
        for (x, y) in positions {
            vps.push(Particle {
                position: (x, y), velocity: (0.0, 0.0),
                acceleration: (0.0, 0.0), mass: 1.0,
                trail: vec![], trail_length: 0, t: 0.0,
                lifespan: 0.0, rgba: (0, 0, 0, 0)
            });
        }
        RigidBody {
            virtual_particles: vps
        }
    }

    pub fn update_derivatives(&mut self, fs: &FluidState, pg: &PixelGrid, dt: f32) {
        let mut central_acceleration = (0.0, 0.0);
        let mut central_velocity = (0.0, 0.0);
        let mut c = 0;
        for p in &mut self.virtual_particles {
            update_particle_derivatives(fs, pg, p, dt);
            central_acceleration.0 += p.acceleration.0;   
            central_acceleration.1 += p.acceleration.1;           
            central_velocity.0 += p.velocity.0;   
            central_velocity.1 += p.velocity.1;
            c += 1;
        }
        let mut total_mass = self.virtual_particles.len() as f32;
        for p in &mut self.virtual_particles {
            p.acceleration.0 = central_acceleration.0 / total_mass;
            p.acceleration.1 = central_acceleration.1 / total_mass;
            p.velocity.0 = central_velocity.0 / total_mass;
            p.velocity.1 = central_velocity.1 / total_mass;
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
        // finally, update them all
        for particle in &mut self.virtual_particles {
            particle.position.0 += dx;
            particle.position.1 += dy;
        }
    }
    pub fn evolve(&mut self, fs: &FluidState, pg: &PixelGrid, dt: f32) {
        self.update_derivatives(fs, pg, dt);
        self.update_position(fs, pg, dt);
    }
}

fn update_particle_derivatives(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, dt: f32) {
    // fluid forces, for unit density and unit area, is just velocity
    // this can be calculated from the momentum flux
    // or, imagine water is little balls
    // a flow of z per second collisions is z newtons per second
    // and ofc you can relate mass to density * area 

    // however, what's also important is _relative_ velocity
    // since if our object is going the same velocity as the flow
    // there will be no momentum flux
    let fx_hydro = pg.sample_bilinear(&fs.u, p.get_x(), p.get_y()) - p.get_u();
    let fy_hydro = pg.sample_bilinear(&fs.v, p.get_x(), p.get_y()) - p.get_v();

    p.acceleration.0 += dt * fx_hydro / p.mass; // makes sense, the higher the mass the lower the acceleration
    p.acceleration.1 += dt * fy_hydro / p.mass;
    p.velocity.0 += p.acceleration.0 * dt;
    p.velocity.1 += p.acceleration.1 * dt;
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
    match pg.sample_world(&(fs.boundary), p.get_x() + prop_dx, p.get_y()) {
        b if b > 0.0 => {
            ret.0 = prop_dx;
        }
        _ => { }
    }
    match pg.sample_world(&(fs.boundary), p.get_x(), p.get_y() + prop_dy) {
        b if b > 0.0 => {
            ret.1 = prop_dy;
        }
        _ => { }
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
        pos1.push((1.0, 1.0));
        let mut rb1 = RigidBody::new(pos1);
        let mut p = Particle {
            position: (1.0, 1.0),
            velocity: (0.0,  0.0),
            acceleration: (0.0, 0.0),
            mass: 1.0,
            trail: vec![],
            trail_length: 0,
            t: 0.0,
            lifespan: 4.0,
            rgba: (0, 0, 0, 0)        
        };
        for _it in 0..6 {
            evolve_particle(&fs, &pg, &mut p, 1.0);
            rb1.evolve(&fs, &pg, 1.0);
        }
        assert!(p.position.0 == 7.0); // started at 1, 1, did 6 steps
        assert!(p.position.1 == 4.0);
        assert!(rb1.virtual_particles[0].get_x() == 7.0);
        assert!(rb1.virtual_particles[0].get_y() == 4.0);

        let mut pos2 = vec![];
        pos2.push((1.0, 1.0));
        pos2.push((1.0, 2.0));
        let mut rb2 = RigidBody::new(pos2);
        pg.print_data(&fs.boundary);
        for _it in 0..10 {
            rb1.evolve(&fs, &pg, 1.0);
        }

        assert!(rb2.virtual_particles[0].get_y() == rb2.virtual_particles[1].get_y() - 1.0);

        assert!(rb2.virtual_particles[0].get_x() < 8.0);
        assert!(rb2.virtual_particles[0].get_y() < 7.0);
        assert!(rb2.virtual_particles[1].get_x() < 8.0);
        assert!(rb2.virtual_particles[1].get_y() < 8.0);

    }

    #[test]
    fn test_evolve_particle() {
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
            velocity: (0.0,  0.0),
            acceleration: (0.0, 0.0),
            mass: 1.0,
            trail: vec![],
            trail_length: 0,
            t: 0.0,
            lifespan: 4.0,
            rgba: (0, 0, 0, 0)        
        };
        pg.print_data(&fs.u);
        pg.print_data(&fs.v);
        pg.print_data(&fs.boundary);
        println!("{} {}", p.position.0, p.position.1);

        evolve_particle(&fs, &pg, &mut p, 1.0);
        println!("{} {}", p.position.0, p.position.1);
        assert!(! p.is_dead());
        for _it in 0..5 {
            evolve_particle(&fs, &pg, &mut p, 1.0);
            println!("{} {}", p.position.0, p.position.1);
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
