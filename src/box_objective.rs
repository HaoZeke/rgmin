//! Extend an objective by infinity outside a session's coordinate box.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Error, Result};
use crate::lbfgs_qp::side_at;
use crate::newton::HessianObjective;

pub(crate) struct BoxObjective<'a, O: ?Sized> {
    inner: &'a O,
    bounds: Bounds<f64>,
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> BoxObjective<'a, O> {
    pub(crate) fn new(inner: &'a O, lo: Option<Vec<f64>>, hi: Option<Vec<f64>>) -> Result<Self> {
        let dim = Objective::dim(inner);
        for side in [&lo, &hi].into_iter().flatten() {
            if !side.is_empty() && side.len() != 1 && side.len() != dim {
                return Err(Error::Dim {
                    got: side.len(),
                    dim,
                });
            }
        }
        let mut low = inner.bounds().low.clone();
        let mut high = inner.bounds().high.clone();
        if low.len() != dim || high.len() != dim {
            return Err(Error::Dim {
                got: low.len(),
                dim,
            });
        }
        for k in 0..dim {
            let requested_low = side_at(lo.as_deref(), k).unwrap_or(f64::NEG_INFINITY);
            let requested_high = side_at(hi.as_deref(), k).unwrap_or(f64::INFINITY);
            if !(requested_low <= requested_high) || !(low[k] <= high[k]) {
                return Err(Error::Highs("invalid coordinate box".into()));
            }
            low[k] = low[k].max(requested_low);
            high[k] = high[k].min(requested_high);
            if !(low[k] <= high[k]) {
                return Err(Error::Highs("invalid coordinate box".into()));
            }
        }
        Ok(Self {
            inner,
            bounds: Bounds::new(low, high, 0.0),
        })
    }

    fn limits(&self, k: usize) -> (f64, f64) {
        (self.bounds.low[k], self.bounds.high[k])
    }

    fn contains(&self, x: ArrayView1<f64>) -> bool {
        x.len() == Objective::dim(self.inner)
            && x.iter().enumerate().all(|(k, &value)| {
                let (lo, hi) = self.limits(k);
                value.is_finite() && lo <= value && value <= hi
            })
    }

    pub(crate) fn clip_start(&self, x: &mut Array1<f64>) -> Result<bool> {
        let dim = Objective::dim(self.inner);
        if x.len() != dim {
            return Err(Error::Dim { got: x.len(), dim });
        }
        let mut changed = false;
        for (k, value) in x.iter_mut().enumerate() {
            let (lo, hi) = self.limits(k);
            let projected = value.clamp(lo, hi);
            changed |= projected != *value;
            *value = projected;
        }
        Ok(changed)
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> Objective<f64> for BoxObjective<'_, O> {
    fn dim(&self) -> usize {
        Objective::dim(self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if self.contains(x) {
            self.inner.eval(x)
        } else {
            f64::INFINITY
        }
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> Gradient<f64> for BoxObjective<'_, O> {
    fn dim(&self) -> usize {
        Objective::dim(self.inner)
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        if self.contains(x) {
            self.inner.grad(x)
        } else {
            Array1::from_elem(x.len(), f64::NAN)
        }
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> DifferentiableObjective<f64>
    for BoxObjective<'_, O>
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        if self.contains(x) {
            self.inner.value_and_gradient(x)
        } else {
            (f64::INFINITY, Array1::from_elem(x.len(), f64::NAN))
        }
    }
}

impl<O: HessianObjective + ?Sized> HessianObjective for BoxObjective<'_, O> {
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        if self.contains(x) {
            self.inner.hessian(x)
        } else {
            Array2::from_elem((x.len(), x.len()), f64::NAN)
        }
    }
}
