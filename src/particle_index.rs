//! This module contains indexes for particle neighbor retrieval.

use crate::pixelgrid::PixelGrid;

/// An index for retrieval of neighbors for a given grid cell
pub struct ParticleIndex {
    start2neighbors: Vec<usize>, // an array with particle indices, implicitly sorted into bins
    ak2start: Vec<usize>, // for each ak, gives index aj of nbr_array, with nbrs
    ak2end: Vec<usize>, // not inclusive, like a 0..end, [a, b)
    // TODO: deprecated in favour of 1D case
    pub neighbors: Vec<Vec<usize>>
}

impl ParticleIndex {
    // TODO: reference to pg should be baked in 
    pub fn new(pg: &PixelGrid, n_particles: usize) -> Self {
        ParticleIndex {
            ak2start: vec![0; pg.m * pg.n],
            ak2end: vec![0; pg.m * pg.n],
            start2neighbors: vec![0; n_particles],
            neighbors: vec![vec![]; n_particles]
        }
    }
    /// Update the index given an array of positions
    ///
    /// # Arguments
    ///
    /// * `pg` - A PixelGrid defining a 2D manifold
    ///
    /// * `x` - An array of 2D positions
    ///
    pub fn update(&mut self, pg: &PixelGrid, x: &[(f32, f32)]) {
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

    }
    // TODO: extend to allow more than 9 slices
    /// For a given world-space position, and a grid, get all 9 neighbors in adjacent cells
    ///
    /// # Arguments
    ///
    /// * `pg` - PixelGrid defining 2D manifold
    ///
    /// * `wx` - x position in world space
    /// 
    /// * `wy` - y position in world space
    pub fn get_nbrs_nine_slice<'a>(&'a self, pg: &PixelGrid, wx: f32, wy: f32) -> [&'a [usize]; 9] {
        let mut result: [&[usize]; 9] = [&[]; 9];
        let mut idx = 0;
        // TODO: inefficient? make a const array 
        for dj in -1..=1 {
            for di in -1..=1 {
                let (wxt, wyt) = (wx + di as f32, wy + dj as f32);
                if wxt < pg.x || wxt >= pg.x + pg.w || wyt < pg.y || wyt >= pg.y + pg.h {
                    continue;
                }

                let (x, y) = pg.worldxy2xy(wxt, wyt);
                let ak = pg.xy2ak(x, y);
                let start = self.ak2start[ak];
                if start >= self.start2neighbors.len() {
                    result[idx] = &[];
                } else {
                    let end = self.ak2end[ak];
                    result[idx] = &self.start2neighbors[start..end];
                }
                idx += 1;
            }
        }
        result
    }
    pub fn get_nbrs(&self, pg: &PixelGrid, wx: f32, wy: f32) -> Vec<usize> {
        let slices = self.get_nbrs_nine_slice(pg, wx, wy);      
        slices.iter().flat_map(|slice| slice.iter().copied()).collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle_ecs::*;

    #[test]
    fn test_particle_index_count() {
        let pg = PixelGrid::new(4, 4);
        let n_particles = pg.n * pg.m;
        let mut pdata = ParticleData::new(n_particles, n_particles);

        let mut pi = 0;
        for i in 0..pg.m {
            for j in 0..pg.n {
                pdata.x[pi] = (j as f32, i as f32);
                pi += 1;
            }
        }
        let mut index = ParticleIndex::new(&pg, n_particles);
        index.update(&pg, &pdata.x);
        println!("");
        for i in 0..pg.m {
            for j in 0..pg.n {
                println!("{} {}: ", i, j);
                let nbrs: Vec<usize> = index.get_nbrs(&pg, i as f32, j as f32);
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
        let pg = PixelGrid::new_with_transform(10, 10, 1.0, 1.0, -5.0, -5.0);

        let np = pg.n - 1;
        let mp = pg.m - 1;

        let n_particles = np * mp;
        let mut pdata = ParticleData::new(n_particles, n_particles);

        // arrange the particles on a grid
        for i in 0..mp {
            for j in 0..np {
                pdata.x[i * np + j] = (-5.0 + j as f32 + 0.5, -5.0 + i as f32 + 0.5);
            }
        }

        let mut index = ParticleIndex::new(&pg, n_particles);
        index.update(&pg, &pdata.x);
        // index.update_neighbors(&pg, &pdata.x, 1);

        let nbrs = index.get_nbrs(&pg, -5.0, -5.0);
        assert!(nbrs.len() == 4);
        assert!(nbrs == vec![0, 1, np, np + 1]);
        // assert!(nbrs == index.neighbors[0]);

        let nbrs = index.get_nbrs(&pg, -5.0, -4.0);
        println!("{} {}: {:?}", -5.0, -4.0, nbrs);
        assert!(nbrs.len() == 6);
        assert!(nbrs == vec![0, 1, np, np + 1, 2 * np, 2 * np + 1]);

        let nbrs: Vec<usize> = index.get_nbrs(&pg, -4.0, -5.0);
        println!("{} {}: {:?}", -4.0, -5.0, nbrs);
        assert!(nbrs.len() == 6);
        assert!(nbrs == vec![0, 1, 2, np, np + 1, np + 2]);

        let nbrs: Vec<usize> = index.get_nbrs(&pg, -4.0, -4.0);
        println!("{} {}: {:?}", -4.0, -4.0, nbrs);
        assert!(nbrs.len() == 9);
        assert!(nbrs == vec![0, 1, 2, np, np + 1, np + 2, 2 * np, 2 * np + 1, 2 * np + 2]);

        // now, move one of the particles to the middle
        pdata.x[0] = (0.5, 0.5);
        index.update(&pg, &pdata.x);
        // index.update_neighbors(&pg, &pdata.x, 1);

        let nbrs: Vec<usize> = index.get_nbrs(&pg, 0.5, 0.5);
        assert!(nbrs.len() == 10);
        println!("{} {}: {:?}", 0.5, 0.5, nbrs);
        assert!(nbrs.contains(&0));

        let nbrs: Vec<usize> = index.get_nbrs(&pg, 1.5, 1.5);
        assert!(nbrs.len() == 10);
        println!("{} {}: {:?}", 1.5, 1.5, nbrs);
        assert!(nbrs.contains(&0));

        // Check that nine_slice does the same thing.
        let slices = index.get_nbrs_nine_slice(&pg, 1.5, 1.5);
        let mut res = vec![];
        for slice in slices.iter() {
            for &ind in *slice {
                res.push(ind);
                assert!(nbrs.contains(&ind));
            }
        }
        assert!(res.len() == nbrs.len());

        // Check that nine_slice works for edges
        let slices = index.get_nbrs_nine_slice(&pg, -5.0, -5.0);
        let mut res = vec![];
        for slice in slices.iter() {
            for &ind in *slice {
                res.push(ind);
            }
        }
        println!("{:?}", res);
        assert!(res == vec![1, np, np + 1]); // 1 was moved

    }
} 
