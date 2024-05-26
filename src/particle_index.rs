use crate::pixelgrid::PixelGrid;
use crate::particle::Particle;
use crate::particle_ecs::*;
use std::sync::Arc;

pub struct ParticleIndex {
    start2neighbors: Vec<usize>, // an array with particle indices, implicitly sorted into bins
    ak2start: Vec<usize>, // for each ak, gives index aj of nbr_array, with nbrs
    ak2end: Vec<usize>, // not inclusive, like a 0..end, [a, b)
    pub neighbors: Vec<Vec<usize>>
}

impl ParticleIndex {
    pub fn new(pg: &PixelGrid, n_particles: usize) -> Self {
        ParticleIndex {
            ak2start: vec![0; pg.m * pg.n],
            ak2end: vec![0; pg.m * pg.n],
            start2neighbors: vec![0; n_particles],
            neighbors: vec![vec![]; n_particles]
        }
    }
    pub fn update(&mut self, pg: &PixelGrid, x: &Vec<(f32, f32)>) {
        let mut pi2ak = vec![0; x.len()];
        let mut pi2ak_sorted = vec![0; x.len()];
        self.start2neighbors = (0..pi2ak.len()).collect();
        for pi in 0..x.len() {
            let (x, y) = pg.worldxy2xy(x[pi].0, x[pi].1);
            let ak = pg.xy2ak(x, y);
            pi2ak[pi] = ak;
        }
        // piarr:                   [0, 1, 2, 3, 4]
        // pi2ak:                   [1, 3, 5, 4, 1]
        // pi2ak_sorted:            [1, 1, 3, 4, 5]
        // start2neighbors:         [0, 4, 1, 3, 2] // start2nbrs is also just argsort of pi2ak
        // ak2start:                [N, 0, N, 2, 3, 4]
        // ak2end:                  [N, 2, N, 3, 4, 5]
        
        self.start2neighbors.sort_by(|&i, &j| pi2ak[i].cmp(&pi2ak[j]));
        for (pi, ind) in self.start2neighbors.iter().enumerate() {
            pi2ak_sorted[pi] = pi2ak[*ind];
        }
        self.ak2start = vec![pi2ak_sorted.len() + 2; pg.m * pg.n]; // out of bounds = NaN
        self.ak2end = vec![pi2ak_sorted.len() + 2; pg.m * pg.n];
        for (i, ak) in pi2ak_sorted.iter().enumerate() {
            if self.ak2start[*ak] == pi2ak_sorted.len() + 2 {
                self.ak2start[*ak] = i; // only do this if it's not already been set
            }
        }
        for (i, ak) in pi2ak_sorted.iter().enumerate().rev() {
            if self.ak2end[*ak] == pi2ak_sorted.len() + 2 {
                assert!(self.ak2start[*ak] < pi2ak_sorted.len() + 2); // must have a start
                self.ak2end[*ak] = i + 1; // only do this if it's not already been set
            }
        }

        // let piarr: Vec<usize> = (0..pi2ak.len()).collect();
        // println!("{:?}", piarr);
        // println!("{:?}", pi2ak);
        // println!("{:?}", pi2ak_sorted);
        // println!("{:?}", self.ak2start);
        // println!("{:?}", self.ak2end);
    }
    pub fn get_nbrs(&self, pg: &PixelGrid, wx: f32, wy: f32, dist: i32) -> Vec<usize> {
        let mut result = vec![];
        for di in -dist..dist+1 {
            for dj in -dist..dist+1 {
                let (x, y) = pg.worldxy2xy(wx + di as f32, wy + dj as f32);
                if (x < pg.x || x >= pg.x + pg.w || y < pg.y || y >= pg.h) {
                    continue;
                }
                let ak = pg.xy2ak(x, y);
                let start = self.ak2start[ak];
                let end = self.ak2end[ak];
                // println!("({} {}) -> ({} {}) -> ({} {}) -> {} -> {} -> {}", wx, wy, wx + di as f32, wy + dj as f32, x, y, ak, start, end);
                for k in start..end {
                    let nbrj = self.start2neighbors[k];
                    result.push(nbrj);
                }
            }
        }
        result
    }
    pub fn update_neighbors(&mut self, pg: &PixelGrid, x: &Vec<(f32, f32)>, dist: i32) {
        for pi in 0..x.len() {
            self.neighbors[pi].clear();
            let (x, y) = pg.worldxy2xy(x[pi].0, x[pi].1);
            let mut nbrs = self.get_nbrs(pg, x, y, dist);
            self.neighbors[pi].append(&mut nbrs);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;


#[test]
    fn test_particle_index_count() {
        let pg = PixelGrid::new(5, 5);
        let n_particles = (pg.n - 2) * (pg.m - 2);
        let mut pdata = ParticleData::new(n_particles, n_particles);

        let mut pi = 0;
        for i in 0..pg.m-2 {
            for j in 0..pg.n-2 {
                pdata.x[pi] = (j as f32 + 1.0 as f32, i as f32 + 1.0);
                pi += 1;
            }
        }
        let mut index = ParticleIndex::new(&pg, n_particles);
        index.update(&pg, &pdata.x);
        println!("");
        for i in 1..pg.m-1 {
            for j in 1..pg.n-1 {
                println!("{} {}: ", i, j);
                let nbrs = index.get_nbrs(&pg, i as f32, j as f32, 1);
                println!("{:?}", nbrs);
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

        let n_particles = pg.n * pg.m;
        let mut pdata = ParticleData::new(n_particles, n_particles);

        for i in 0..pg.m {
            for j in 0..pg.n {
                pdata.x[i * pg.n + j] = (j as f32, i as f32);
            }
        }
        let mut index = ParticleIndex::new(&pg, n_particles);
        index.update(&pg, &pdata.x);

        println!("");
        for i in 0..pg.m {
            for j in 0..2 {
                let nbrs = index.get_nbrs(&pg, i as f32, j as f32, 1);
                println!("{} {}: {} ", i, j, nbrs.len());
                assert!(nbrs.len() == 0);
            }
        }
        println!("");
        for i in 0..pg.m {
            for j in 8..10 {
                let nbrs = index.get_nbrs(&pg, i as f32, j as f32, 1);
                println!("{} {}: {} ", i, j, nbrs.len());
                assert!(nbrs.len() == 0);
            }
        }
        println!("");
        for i in 5..7 {
            for j in 5..7 {
                let nbrs = index.get_nbrs(&pg, i as f32, j as f32, 1);
                assert!(nbrs.len() == 2);
                println!("{} {}: {} ", i, j, nbrs.len());
            }
        }
    }

}