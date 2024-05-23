use crate::pixelgrid::PixelGrid;
use crate::particle::Particle;
use crate::particle_ecs::*;

pub struct ParticleIndex {
    slots: Vec<Vec<usize>>,
    pub neighbors: Vec<Vec<usize>>
}

impl ParticleIndex {
    pub fn new(pg: &PixelGrid) -> Self {
        let mut slots = vec![];
        for _ in 0..pg.mn {
            slots.push(vec![]);
        }
        ParticleIndex {
            slots: slots,
            neighbors: vec![vec![]]
        }
    }
    pub fn clear(&mut self) {
        for ak in 0..self.slots.len() {
            self.slots[ak].clear();
        }
        self.neighbors.clear();
    }
    pub fn update(&mut self, pg: &PixelGrid, particles: &Vec<Particle>) {
        self.clear();
        for pi in 0..particles.len() {
            let (x, y) = pg.worldxy2xy(particles[pi].get_x(), particles[pi].get_y());
            let ak = pg.xy2ak(x, y);
            self.slots[ak].push(pi);
        }
    }
    pub fn update_ecs(&mut self, pg: &PixelGrid, pdata: &ParticleData) {
        self.clear();
        for pi in 0..pdata.n_particles {
            let (x, y) = pg.worldxy2xy(pdata.x[pi].0, pdata.x[pi].1);
            let ak = pg.xy2ak(x, y);
            self.slots[ak].push(pi);
        }
    }
    pub fn update_neighbors(&mut self, pg: &PixelGrid, pdata: &ParticleData, dist: f32) {
        for pi in 0..pdata.n_particles {
            let (x, y) = pdata.x[pi];
            self.neighbors.push(self.get_nbrs(pg, x, y, dist));
        }
    }
    pub fn get_nbrs(&self, pg: &PixelGrid, wx: f32, wy: f32, dist: f32) -> Vec<usize> {
        let mut result = vec![];
        let (x, y) = pg.worldxy2xy(wx, wy);
        let i0 = (y - dist).max(0.0) as usize;
        let ie = (y + dist + 1.0).min(pg.m as f32) as usize;
        let j0 = (x - dist).max(0.0) as usize;
        let je = (x + dist + 1.0).min(pg.n as f32) as usize;

        for y2 in i0..ie {
            for x2 in j0..je {
                let ak = pg.xy2ak(x2 as f32, y2 as f32);
                for ind in &self.slots[ak] {
                    result.push(*ind);
                }
            }
        }
        
        return result;
    }
}

pub fn cull_nbrs(pi: usize, particles: &mut Vec<Particle>, h: f32) {
    let mut new_nbrs = vec![];
    for nbrj in &particles[pi].nbrs {
        if particles[pi].dist(&particles[*nbrj]) <= h{
            new_nbrs.push(*nbrj);
        }
    }
    particles[pi].nbrs = new_nbrs;
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



#[cfg(test)]
mod tests {
    use super::*;

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