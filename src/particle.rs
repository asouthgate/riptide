use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;

#[derive(Default)]
struct Particle {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    x: f32,
    y: f32,
    mass: f32,
    trail: Vec<(f32, f32)>,
    t: f32,
    lifespan: f32
}

impl Particle {
    pub fn is_dead(&self) -> bool{
        self.t > self.lifespan
    }
}

fn evolve_particle(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, dt: f32) {
    p.t += dt;
    let u = pg.sample_bilinear(&fs.u, p.x, p.y);
    let v = pg.sample_bilinear(&fs.v, p.x, p.y);
    println!("  {} {}", u, v);
    let prop_px = p.x + u * dt * p.mass;
    let prop_py = p.y + v * dt * p.mass;
    match pg.sample_world(&(fs.boundary), prop_px, p.y) {
        b if b > 0.0 => {
            p.x = prop_px;
        }
        _ => {}
    }
    match pg.sample_world(&(fs.boundary), p.x, prop_py) {
        b if b > 0.0 => {
            p.y = prop_py;
        }
        _ => {}
    }
}

fn evolve_particle_trail(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, dt: f32) {
    evolve_particle(fs, pg, p, dt);
    if p.trail.len() > 0 {
        let (lastx, lasty) = p.trail[p.trail.len()-1];
        if lastx.floor() != p.x.floor() || lasty.floor() != p.y.floor() {
            p.trail.push((p.x, p.y));
        }
    } else {
        p.trail.push((p.x, p.y));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            r: 0,
            g: 0,
            b: 0,
            a: 0,
            x: 1.0,
            y: 1.0,
            mass: 1.0,
            trail: vec![],
            t: 0.0,
            lifespan: 4.0        
        };
        pg.print_data(&fs.u);
        pg.print_data(&fs.v);
        pg.print_data(&fs.boundary);
        println!("{} {}", p.x, p.y);

        evolve_particle_trail(&fs, &pg, &mut p, 1.0);
        println!("{} {}", p.x, p.y);
        assert!(! p.is_dead());
        for _it in 0..5 {
            evolve_particle_trail(&fs, &pg, &mut p, 1.0);
            println!("{} {}", p.x, p.y);
        }
        assert!(p.is_dead());
        assert!(p.x == 7.0); // started at 1, 1, did 6 steps
        assert!(p.y == 4.0);

        // now test the boundary
        for _it in 0..50 {
            evolve_particle_trail(&fs, &pg, &mut p, 1.0);
            println!("{} {}", p.x, p.y);
        }
        assert!(p.x <= 8.0);
        assert!(p.y <= 8.0);
    }

}