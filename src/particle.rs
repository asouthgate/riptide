use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;

struct Particle {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    x: f32,
    y: f32,
    mass: f32,
    trail: Vec<(f32, f32)>
}

fn evolve_particle(fs: &FluidState, pg: &PixelGrid, p: &mut Particle, dt: f32) {
    let u = pg.sample_bilinear(&fs.u, p.x, p.y);
    let v = pg.sample_bilinear(&fs.v, p.x, p.y);
    p.x += u * dt * p.mass;
    p.y += v * dt * p.mass;
    if p.trail.len() > 0 {
        let (lastx, lasty) = p.trail[p.trail.len()-1];
        if (lastx.floor() != p.x.floor() || lasty.floor() != p.y.floor()) {
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

    }

}