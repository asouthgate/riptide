use std::ops::{
    Add, AddAssign, Sub, 
    SubAssign, Mul, MulAssign, 
    Div, DivAssign, Index,
    IndexMut
};

pub trait Vector<T>: 
    Add<Output = Self> + Sub<Output = Self> + 
    Mul<T, Output = Self> + Div<T, Output=Self> + 
    Sized + 
    Index<usize, Output=T> + 
    IndexMut<usize> + 
    MulAssign<T> + 
    DivAssign<T> +
    AddAssign<Self> +
    SubAssign<Self> +
    Copy +
    Send + 
    Sync
{
    fn magnitude(&self) -> T;
    fn dot(&self, other: &Self) -> T;
}


impl Vector<f32> for cgmath::Vector2<f32> {
    fn magnitude(&self) -> f32 {
        self.dot(self).sqrt()
    }
    fn dot(&self, other: &cgmath::Vector2<f32>) -> f32 {
        self.x * other.x + self.y * other.y
    }
}


impl Vector<f32> for cgmath::Vector3<f32> {
    fn magnitude(&self) -> f32 {
        self.dot(self).sqrt()
    }
    fn dot(&self, other: &cgmath::Vector3<f32>) -> f32 {
        self.x * other.x + self.y * other.y
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Vector2, Vector3};

    fn _test_trait_mutref<V: Vector<f32>>(v: &mut V) -> bool {
        *v *= 0.5;
        *v = *v / 2.0;
        v[0] += 0.0;
        v[1] *= 0.0;

        let mut w = *v;
        w[0] = 1.0;
        w[1] = 1.9;
        true
    }

    #[test]
    fn test_cgmath() {
        let mut v1: Vector2<f32> = Vector2::new(1.0, 0.0);
        _test_trait_mutref(&mut v1);
        let mut v2: Vector3<f32> = Vector3::new(1.0, 0.0, 0.0);
        _test_trait_mutref(&mut v2);
        println!("{} {}", v1.x, v1.y);
        assert!(v1.x == 0.25);
        assert!(v2.x == 0.25);
    }
}