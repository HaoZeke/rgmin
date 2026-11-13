//! `Lbfgs::minimize_box` against SciPy 1.18 L-BFGS-B on bound-constrained
//! problems. The coupled quadratic is not the projection of its free
//! minimum, and the Rosenbrock minimum sits on a bound.

use ndarray::{Array1, ArrayView1};
use rgmin::Lbfgs;

const GTOL: f64 = 1e-8;
const MEMORY: usize = 8;

fn scipy(problem: &str) -> (Vec<f64>, f64) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/lbfgsb_scipy_gold.py");
    let py = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let payload =
        format!(r#"{{"problem":"{problem}","maxcor":{MEMORY},"gtol":{GTOL},"ftol":1e-20}}"#);
    let mut child = Command::new(&py)
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("{py} spawn: {e}"));
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(payload.as_bytes())
        .expect("write gold stdin");
    let out = child.wait_with_output().expect("gold wait");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "scipy gold failed: {text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("\"success\": true") || text.contains("\"success\":true"),
        "scipy did not converge: {text}"
    );
    let x = json_array(&text, "\"x\"");
    let fun = json_number(&text, "\"fun\"");
    (x, fun)
}

fn json_array(text: &str, key: &str) -> Vec<f64> {
    let start = text
        .find(key)
        .unwrap_or_else(|| panic!("missing {key} in {text}"));
    let lb = text[start..].find('[').expect("[") + start;
    let rb = text[lb..].find(']').expect("]") + lb;
    text[lb + 1..rb]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("number"))
        .collect()
}

fn json_number(text: &str, key: &str) -> f64 {
    let start = text
        .find(key)
        .unwrap_or_else(|| panic!("missing {key} in {text}"));
    let colon = text[start..].find(':').expect(":") + start;
    let rest = text[colon + 1..].trim_start();
    let end = rest.find([',', '}']).expect("end");
    rest[..end].trim().parse().expect("number")
}

fn same_point(problem: &str, got: &Array1<f64>, got_f: f64, want: &[f64], want_f: f64) {
    assert_eq!(got.len(), want.len(), "{problem}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "{problem} x[{i}]: minimize_box {a} scipy {b}"
        );
    }
    assert!(
        (got_f - want_f).abs() < 1e-6,
        "{problem} f: minimize_box {got_f} scipy {want_f}"
    );
}

fn active(x: f64, lo: f64, hi: f64) -> bool {
    (x - lo).abs() <= 1e-5 || (x - hi).abs() <= 1e-5
}

fn separable(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let scales = [1.0, 4.0, 9.0, 16.0, 0.5, 2.0];
    let targets = [2.5, -1.0, 0.3, 8.0, 1.2, -3.0];
    let g = Array1::from_shape_fn(x.len(), |i| scales[i] * (x[i] - targets[i]));
    let f = 0.5
        * (0..x.len())
            .map(|i| scales[i] * (x[i] - targets[i]).powi(2))
            .sum::<f64>();
    (f, g)
}

fn separable_box() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    (
        Array1::from(vec![0.5, 0.0, 0.0, 1.0, 0.0, 0.0]),
        Array1::from(vec![0.0, -2.0, -1.0, 0.0, -5.0, -1.0]),
        Array1::from(vec![1.0, 2.0, 1.0, 3.0, 5.0, 0.5]),
    )
}

fn coupled(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let t = [3.0, -2.0, 0.2, 4.0, -3.0];
    let n = x.len();
    let apply = |v: &[f64]| {
        let mut hv = vec![0.0; n];
        for i in 0..n {
            hv[i] = 4.0 * v[i];
            if i > 0 {
                hv[i] += v[i - 1];
            }
            if i + 1 < n {
                hv[i] += v[i + 1];
            }
        }
        hv
    };
    let hx = apply(x.as_slice().expect("contiguous"));
    let b = apply(&t);
    let g = Array1::from_shape_fn(n, |i| hx[i] - b[i]);
    let f = 0.5 * x.dot(&Array1::from(hx)) - x.dot(&Array1::from(b));
    (f, g)
}

fn coupled_box() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    (
        Array1::zeros(5),
        Array1::from(vec![0.0, -1.0, -2.0, 0.0, -1.0]),
        Array1::from(vec![1.5, 1.0, 2.0, 1.0, 1.0]),
    )
}

fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (a, b) = (x[0], x[1]);
    let t = b - a * a;
    (
        (1.0 - a).powi(2) + 100.0 * t * t,
        Array1::from(vec![-2.0 * (1.0 - a) - 400.0 * a * t, 200.0 * t]),
    )
}

fn rosenbrock_box() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    (
        Array1::from(vec![-1.2, 1.0]),
        Array1::from(vec![-2.0, -1.0]),
        Array1::from(vec![0.5, 2.0]),
    )
}

fn relax(
    fg: impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>),
    x0: ArrayView1<f64>,
    lo: ArrayView1<f64>,
    hi: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    let mut opt = Lbfgs::with_capacity(MEMORY);
    opt.gtol = GTOL;
    let (f, x, _) = opt.minimize_box(x0, lo, hi, 500, |v| Some(fg(v)));
    (f, x)
}

#[test]
fn separable_quadratic_matches_scipy_with_three_active_bounds() {
    let (x0, lo, hi) = separable_box();
    let (f, x) = relax(separable, x0.view(), lo.view(), hi.view());
    let (want, want_f) = scipy("separable");
    same_point("separable", &x, f, &want, want_f);
    let bound: Vec<usize> = (0..x.len())
        .filter(|&i| active(want[i], lo[i], hi[i]))
        .collect();
    assert_eq!(bound, vec![0, 3, 5], "scipy active set {bound:?}");
    for i in bound {
        assert!(
            active(x[i], lo[i], hi[i]),
            "x[{i}] = {} left the bound",
            x[i]
        );
    }
}

#[test]
fn coupled_quadratic_matches_scipy_off_the_clipped_free_minimum() {
    let (x0, lo, hi) = coupled_box();
    let (f, x) = relax(coupled, x0.view(), lo.view(), hi.view());
    let (want, want_f) = scipy("coupled");
    same_point("coupled", &x, f, &want, want_f);
    // The free minimizer's third coordinate is 0.2. Coupling moves it.
    assert!(
        (x[2] - 0.2).abs() > 0.1,
        "x[2] = {} is the clipped free minimum",
        x[2]
    );
    let bound: Vec<usize> = (0..x.len())
        .filter(|&i| active(want[i], lo[i], hi[i]))
        .collect();
    assert!(bound.len() >= 2, "expected active bounds, scipy x {want:?}");
    for i in bound {
        assert!(
            active(x[i], lo[i], hi[i]),
            "x[{i}] = {} left the bound",
            x[i]
        );
    }
}

#[test]
fn bounded_rosenbrock_matches_scipy_on_the_active_face() {
    let (x0, lo, hi) = rosenbrock_box();
    let (f, x) = relax(rosenbrock, x0.view(), lo.view(), hi.view());
    let (want, want_f) = scipy("rosenbrock");
    same_point("rosenbrock", &x, f, &want, want_f);
    assert!(
        active(want[0], lo[0], hi[0]),
        "scipy left the bound {want:?}"
    );
    assert!(active(x[0], lo[0], hi[0]), "x[0] = {}", x[0]);
    assert!(
        (x[0] - 0.5).abs() < 1e-5 && (x[1] - 0.25).abs() < 1e-5,
        "{x}"
    );
}
