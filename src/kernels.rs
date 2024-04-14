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

pub fn debrun_spiky_kernel(r: f32, h: f32) -> f32 {
    match r {
        _r if r < 0.0 => {
            0.0
        },
        _r if r > h => {
            0.0
        }
        _ => {
            let coeff = 10.0 / (PI * h.powi(5)); // TWO DIMENSIONAL CASE
            coeff * (h - r).powi(3)
        }
    }
}

pub fn debrun_spiky_kernel_dwdr(r: f32, h: f32) -> f32 {
    match r {
        _r if r < 0.0 => {
            0.0
        },
        _r if r > h => {
            0.0
        }
        _ => {
            let coeff = 10.0 / (PI * h.powi(6));
            - 3.0 * coeff * (h - r).powi(2)
        }
    }
}

pub fn cal_r(dx: f32, dy: f32) -> f32 {
    (dx.powi(2) + dy.powi(2)).powf(0.5)
}

pub fn debrun_spiky_kernel_grad(dx: f32, dy: f32, h: f32) -> (f32, f32) {
    let r = cal_r(dx, dy);
    let dwdror = debrun_spiky_kernel_dwdr(r, h) / r;
    (dx * dwdror, dy * dwdror)
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

    #[test]
    fn test_spiky_kernel() {
        let h: f32 = 1.329;
        let r: f32 = 0.39881;
        assert!((debrun_spiky_kernel(r, h) - 0.6974394347109141).abs() < 0.0000001);
        assert!(debrun_spiky_kernel(-0.000001, h) == 0.0);
        assert!(debrun_spiky_kernel(1.33, h) == 0.0);
    }

    #[test]
    fn test_kernel_dwdr() {
        let h: f32 = 1.329;
        let r: f32 = 0.39881;
        assert!(debrun_spiky_kernel_dwdr(0.0, h) < -4.5); // doesn't disappear at origin
    }

    #[test]
    fn test_kernel_grad() {
        let h: f32 = 1.329;
        let dx: f32 = 0.1361;
        let dy: f32 = 0.9981;
        let r: f32 = cal_r(dx, dy);
        let h = 1.8;
        let grad = debrun_spiky_kernel_grad(dx, dy, h);
        assert!((grad.0 - dx * debrun_spiky_kernel_dwdr(r, h) / r).abs() < 0.000001);
        assert!((grad.1 - dy * debrun_spiky_kernel_dwdr(r, h) / r).abs() < 0.000001);
        assert!(grad.0 < 0.0); // we expect the gradient to be pointing downwards
        assert!(grad.1 < 0.0);
    }

}