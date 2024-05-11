use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;
use crate::particle::*;

#[derive(Default)]
pub struct RigidBody {
    virtual_particles: Vec<Particle>,
    central_particle: Particle,
    relative_positions: Vec<(f32, f32)>,
}

impl RigidBody {
    pub fn new(x0: f32, y0: f32, mass_per_pixel: f32, drag_per_pixel: f32, positions: Vec<(f32, f32)>) -> Self {
        let mut vps = vec![];
        // let n_vps = positions.len() as f32;
        for (x, y) in &positions {
            vps.push(Particle {
                position: (x0 + *x, y0 + *y), 
                mass: mass_per_pixel,
                cdrag: drag_per_pixel,
                ..Default::default()
            });
        }
        RigidBody {
            virtual_particles: vps,
            central_particle: Particle { 
                position: (x0, y0), 
                mass: mass_per_pixel, 
                cdrag: drag_per_pixel,
                ..Default::default() 
            },
            relative_positions: positions,
        }
    }

    pub fn update_derivatives(&mut self, fs: &FluidState, pg: &PixelGrid, dt: f32) {
        self.central_particle.f_hydro = (0.0, 0.0);
        self.central_particle.f_drag = (0.0, 0.0);
        for p in &mut self.virtual_particles {
            update_fluid_forces(fs, pg, p, dt);
            self.central_particle.f_hydro.0 += p.f_hydro.0;
            self.central_particle.f_hydro.1 += p.f_hydro.1;
            self.central_particle.f_drag.0 += p.f_drag.0;
            self.central_particle.f_drag.1 += p.f_drag.1;
        }
        self.central_particle.f_hydro.0 /= self.virtual_particles.len() as f32;
        self.central_particle.f_hydro.1 /= self.virtual_particles.len() as f32;
        self.central_particle.f_drag.0 /= self.virtual_particles.len() as f32;
        self.central_particle.f_drag.1 /= self.virtual_particles.len() as f32;

        update_particle_derivatives(&mut self.central_particle, dt);
        self.central_particle.velocity.0 += self.central_particle.vboost.0;
        self.central_particle.velocity.1 += self.central_particle.vboost.1;
        self.central_particle.vboost = (0.0, 0.0);

        for p in &mut self.virtual_particles {
            p.acceleration.0 = self.central_particle.acceleration.0;
            p.acceleration.1 = self.central_particle.acceleration.1;
            p.velocity.0 = self.central_particle.velocity.0;
            p.velocity.1 = self.central_particle.velocity.1;
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
        for particle in &mut self.virtual_particles {
            let (pdx, pdy) = cal_proposed_step(fs, pg, particle, dt);
            if dx != 0.0 {
                dx = pdx;
            }
            if dy != 0.0 {
                dy = pdy;
            }
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

    pub fn boost(&mut self, bx: f32, by: f32) {
        self.central_particle.vboost.0 += bx;
        self.central_particle.vboost.1 += by;
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_rigidbody_doesnt_get_stuck() {
        let pg = PixelGrid::new(20, 4);  
        let mut fs = FluidState::new(&pg);
        let seed_value = 42;
        let rng = StdRng::seed_from_u64(seed_value);
        for j in 0..pg.n {
            for i in 0..pg.m {
                let ak = i * pg.n + j;
                fs.u[ak] = 0.0;
                fs.v[ak] = 1.0;
            }
        }
        pg.print_data(&fs.u);
        pg.print_data(&fs.v);
        pg.print_data(&fs.boundary);
        let player_rb_positions: Vec<(f32, f32)> = vec![
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ];
        let mut player_rb = RigidBody::new(1.0, 1.0, 1.0, 1.0, player_rb_positions);

        for _it in 0..50 {
            player_rb.evolve(&fs, &pg, 1.0);
            player_rb.central_particle.print();
        }
        assert!(*player_rb.get_y() == 17.0);
        for _it in 0..50 {
            player_rb.evolve(&fs, &pg, 1.0);
            player_rb.boost(0.0, -2.0);
            player_rb.central_particle.print();
        }
        assert!(*player_rb.get_y() == 1.0);
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
        let mut rb1 = RigidBody::new(1.0, 1.0, 1.0, 1.0, pos1);
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
        let mut rb2 = RigidBody::new(1.0, 1.0, 1.0, 1.0, pos2);
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

}