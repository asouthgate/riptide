const PI: f32 = 3.141592653589793;

fn cubic_spline_fac(h: f32) -> f32 {
    10.0 / (7.0 * PI * h.powi(2))
}

pub fn cubic_spline_kernel(r: f32, h: f32) -> f32 {
    let norm = cubic_spline_fac(h);
    let q = r / h;
    let fq = match q {
        _q if _q > 2.0 => 0.0,
        _q if _q > 1.0 => 0.25 * (2.0 - q).powi(3),
        _ => 1.0 - 1.5 * q.powi(2) * (1.0 - 0.5 * q)
    };
    return norm * fq;
}

fn cubic_spline_kernel_dwdq(r: f32, h: f32) -> f32 {
    let norm = cubic_spline_fac(h);
    let q = r / h;
    let fq = match q {
        _q if _q > 2.0 => 0.0,
        _q if _q > 1.0 => -0.75 * (2.0 - q).powi(2),
        _ => - 3.0 * q * (1.0 - 0.75 * q)
    };
    return norm * fq;
}

pub fn cubic_spline_grad(dx: f32, dy: f32, r: f32, h: f32) -> (f32, f32) {
    match r {
        _r if _r > 1e-13 => {
            let dwdq = cubic_spline_kernel_dwdq(r, h);
            let rhinv = 1.0 / (r * h);
            (
                rhinv * dx * dwdq,
                rhinv * dy * dwdq,
            )        
        }
        _ => (0.0, 0.0)
    }
}

fn gaussian_power_kernel_dwdr(r: f32, h: f32) -> f32 {
    0.0
}

pub fn gaussian_power_kernel_grad(dx: f32, dy: f32, h: f32) -> (f32, f32) {
    // dwdx = dwdr * drdx, r = sqrt(x^2 + y^2) -> drdx = x/r
    let r = (dx.powi(2) + dy.powi(2)).powf(0.5);
    let l = gaussian_power_kernel_dwdr(r, h) / r;
    (dx * l, dy * l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};


    #[test]
    fn test_kernel() {
        assert!(cubic_spline_kernel(0.00, 1.0) == 1.0 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel(0.50, 1.0) == 0.71875 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel(1.00, 1.0) == 0.25 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel(1.50, 1.0) == 0.03125 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel(2.00, 1.0) == 0.0);
        assert!(cubic_spline_kernel(2.01, 1.0) == 0.0);
        assert!(cubic_spline_kernel(10.0, 1.0) == 0.0);
    }

    #[test]
    fn test_cubic_spline_kernel_dwdq_unit_h() {
        assert!(cubic_spline_kernel_dwdq(0.00, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(0.50, 1.0) == -0.9375 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(1.00, 1.0) == -0.75 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(1.50, 1.0) == -0.1875 * cubic_spline_fac(1.0));
        assert!(cubic_spline_kernel_dwdq(2.00, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(2.01, 1.0) == 0.0);
        assert!(cubic_spline_kernel_dwdq(10.0, 1.0) == 0.0);
    }

    // #[test]
    // fn test_cubic_spline_kernel_dwdq_bigh() {
    //     let h = 1.0;
    //     assert!(cubic_spline_kernel_dwdq(0.00, h) == 0.0);
    //     let mut xs = vec![];
    //     let maxk = 100;
    //     for k in 0..maxk {
    //         xs.push(h * 2.0 * ( k as f32 / maxk as f32));
    //     }
    //     let mut ys = vec![];
    //     for x in xs {
    //         let y = cubic_spline_kernel_dwdq(x, h);
    //         println!("? {:.5} {:.5} ", x, y);
    //         ys.push(y);
    //     }
    //     println!("")
    //     // println!("{} {} {} {} {}", v0, v05, v1, v, v4, v5);
    // }

    #[test]
    fn test_kernel_lap() {

    }

}