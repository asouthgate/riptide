use crate::vector::{Vector};
use cgmath::{Vector2};

/// An Entity Component System for particle simulation
///
/// This is easy to parallelize, has good locality, and
/// is easier to convert to GPU than OO implementation. 
#[derive(Clone)]
pub struct ParticleData<V: Vector<f32>> {
    pub x: Vec<V>,
    pub v: Vec<V>,
    pub a: Vec<V>,
    pub pressure: Vec<f32>,
    pub density: Vec<f32>,
    pub mass: Vec<f32>,
    pub f_pressure: Vec<V>,
    pub f_viscous: Vec<V>,
    pub f_body: Vec<V>,
    pub f_surface: Vec<V>,
    pub particle_type: Vec<usize>,
    pub n_particles: usize,
    pub n_fluid_particles: usize
}


pub struct ParticleRef<'a, V: Vector<f32>> {
    pub x: &'a mut V,
    pub v: &'a mut V,
    pub a: &'a mut V,
    pub pressure: &'a mut f32,
    pub density: &'a mut f32,
    pub mass: &'a mut f32, 
    pub f_pressure: &'a mut V,
    pub f_viscous: &'a mut V,
    pub f_body: &'a mut V,
    pub f_surface: &'a mut V,
    pub particle_type: &'a mut usize,
}


impl ParticleData<Vector2<f32>> {
    pub fn new(n_particles: usize, n_fluid_particles: usize) -> Self {
        ParticleData {
            x: vec![Vector2::new(0.0, 0.0); n_particles],
            v: vec![Vector2::new(0.0, 0.0); n_particles],
            a: vec![Vector2::new(0.0, 0.0); n_particles],
            pressure: vec![0.0; n_particles],
            density: vec![1.0; n_particles],
            mass: vec![1.0; n_particles],
            f_pressure: vec![Vector2::new(0.0, 0.0); n_particles],
            f_viscous: vec![Vector2::new(0.0, 0.0); n_particles],
            f_body: vec![Vector2::new(0.0, 0.0); n_particles],
            f_surface: vec![Vector2::new(0.0, 0.0); n_particles],
            particle_type: vec![0; n_particles],
            n_particles: n_particles,
            n_fluid_particles: n_fluid_particles
        }
    }
    pub fn get_particle_ref(&mut self, pid: usize) -> ParticleRef<Vector2<f32>> {
        ParticleRef {
            x: &mut self.x[pid],
            v: &mut self.v[pid],
            a: &mut self.a[pid],
            pressure: &mut self.pressure[pid],
            density: &mut self.density[pid],
            mass: &mut self.mass[pid],
            f_pressure: &mut self.f_pressure[pid],
            f_viscous: &mut self.f_viscous[pid],
            f_body: &mut self.f_body[pid],
            f_surface: &mut self.f_surface[pid],
            particle_type: &mut self.particle_type[pid],
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_ref() {
        let mut pd = ParticleData::<Vector2<f32>>::new(20, 10);
        for k in 0..20 {
            pd.x[k] = Vector2::<f32>::new(k as f32, 0.0);
        }
        let pref = pd.get_particle_ref(7);
        *pref.x = Vector2::<f32>::new(99.0, -99.0);
        assert!(pd.x[7] == Vector2::<f32>::new(99.0, -99.0));
    }

}