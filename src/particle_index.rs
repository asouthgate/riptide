use crate::pixelgrid::PixelGrid;
use crate::vector::Vector;

pub struct ParticleIndex {
    start2neighbors: Vec<usize>, // an array with particle indices, implicitly sorted into bins
    ak2start: Vec<usize>, // for each ak, gives index aj of nbr_array, with nbrs
    ak2end: Vec<usize>, // not inclusive, like a 0..end, [a, b)
    pub neighbors: Vec<Vec<usize>>
}

impl ParticleIndex {
    pub fn new(pg: &PixelGrid, n_particles: usize) -> Self {
        ParticleIndex {
            ak2start: vec![0; pg.size()],
            ak2end: vec![0; pg.size()],
            start2neighbors: vec![0; n_particles],
            neighbors: vec![vec![]; n_particles]
        }
    }
    pub fn update<V: Vector<f32>>(&mut self, pg: &PixelGrid, x: &Vec<V>) {
        let mut pi2ak = vec![0; x.len()];
        let mut pi2ak_sorted = vec![0; x.len()];
        self.start2neighbors = (0..pi2ak.len()).collect();
        for pi in 0..x.len() {
            let (x, y) = pg.worldxy2xy(x[pi][0], x[pi][1]);
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
        self.ak2start = vec![pi2ak_sorted.len() + 2; pg.size()]; // out of bounds = NaN
        self.ak2end = vec![pi2ak_sorted.len() + 2; pg.size()];
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
    pub fn get_nbr_slices<'a>(&'a self, pg: &PixelGrid, wx: f32, wy: f32) -> [&'a [usize]; 9] {
        let mut result: [&[usize]; 9] = [&[]; 9];
        let mut idx = 0;
        for dj in -1..=1 {
            for di in -1..=1 {
                let (wxt, wyt) = (wx + di as f32, wy + dj as f32);
                if pg.out_of_bounds(wxt, wyt, 0.0) {
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
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle_ecs::*;
    use cgmath::Vector2;

    #[test]
    fn test_particle_index_count() {
        let pg = PixelGrid::new(4, 4);
        let n_particles = pg.size();
        let mut pdata = ParticleData::<Vector2<f32>>::new(n_particles, n_particles);

        let mut pi = 0;
        for i in 0..pg.m {
            for j in 0..pg.n {
                pdata.x[pi] = Vector2::<f32>::new(j as f32, i as f32);
                pi += 1;
            }
        }
        let mut index = ParticleIndex::new(&pg, n_particles);
        index.update(&pg, &pdata.x);
        println!("");
        for i in 0..pg.m {
            for j in 0..pg.n {
                println!("{} {}: ", i, j);
                let nbrs: Vec<usize> = index.get_nbr_slices(&pg, i as f32, j as f32)
                    .iter().flat_map(|&inner| inner.iter().cloned()).collect();
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

        let n_particles = pg.size();
        let mut pdata = ParticleData::<Vector2<f32>>::new(n_particles, n_particles);

        // arrange the particles on a grid
        for i in 0..pg.m {
            for j in 0..pg.n {
                pdata.x[i * pg.n + j] = Vector2::<f32>::new(-5.0 + j as f32 + 0.5, -5.0 + i as f32 + 0.5);
            }
        }
        let mut index = ParticleIndex::new(&pg, n_particles);
        index.update(&pg, &pdata.x);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, -5.0, -5.0)
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("{} {}: {:?} {:?}", -5.0, -5.0, nbrs, index.neighbors[0]);
        assert!(nbrs.len() == 4);
        assert!(nbrs == vec![0, 1, 10, 11]);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, -5.0, -4.0)
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("{} {}: {:?}", -5.0, -4.0, nbrs);
        assert!(nbrs.len() == 6);
        assert!(nbrs == vec![0, 1, 10, 11, 20, 21]);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, -4.0, -5.0)
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("{} {}: {:?}", -4.0, -5.0, nbrs);
        assert!(nbrs.len() == 6);
        assert!(nbrs == vec![0, 1, 2, 10, 11, 12]);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, -4.0, -4.0)
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("{} {}: {:?}", -4.0, -4.0, nbrs);
        assert!(nbrs.len() == 9);
        assert!(nbrs == vec![0, 1, 2, 10, 11, 12, 20, 21, 22]);

        // now, move one of the particles to the middle
        pdata.x[0] = Vector2::<f32>::new(0.5, 0.5);
        index.update(&pg, &pdata.x);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, 0.5, 0.5)
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        assert!(nbrs.len() == 10);
        println!("{} {}: {:?}", 0.5, 0.5, nbrs);
        assert!(nbrs.contains(&0));

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, 1.5, 1.5)
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        assert!(nbrs.len() == 10);
        println!("{} {}: {:?}", 1.5, 1.5, nbrs);
        assert!(nbrs.contains(&0));
    }

}