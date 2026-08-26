#include "rgmin/optimize.hpp"
#include "xts/optimize.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

rgmin_status_t quad_eval(void* /*user*/, DLManagedTensorVersioned const* x, double* value) {
    auto const* p = reinterpret_cast<double const*>(
        static_cast<char const*>(x->dl_tensor.data) + x->dl_tensor.byte_offset);
    *value = p[0] * p[0] + p[1] * p[1];
    return RGMIN_SUCCESS;
}

rgmin_status_t quad_grad(void* /*user*/, DLManagedTensorVersioned const* x,
                         DLManagedTensorVersioned* g) {
    auto const* p = reinterpret_cast<double const*>(
        static_cast<char const*>(x->dl_tensor.data) + x->dl_tensor.byte_offset);
    auto* gp = reinterpret_cast<double*>(
        static_cast<char*>(g->dl_tensor.data) + g->dl_tensor.byte_offset);
    gp[0] = 2.0 * p[0];
    gp[1] = 2.0 * p[1];
    return RGMIN_SUCCESS;
}

}  // namespace

int main() {
    std::vector<double> x{1.0, 1.0};
    DLManagedTensorVersioned* t = rgmin::borrow_cpu_f64(x.data(), 2);
    if (t == nullptr) {
        std::fprintf(stderr, "borrow_cpu_f64 returned null\n");
        return 1;
    }
    rgmin::OptimizeControl ctrl;
    ctrl.max_iterations = 80;
    ctrl.gtol = 1e-10;
    rgmin::OptimizeResult r =
        rgmin::optimize(quad_eval, quad_grad, nullptr, t, ctrl, rgmin::Method::Lbfgs);
    if (r.grad_norm > 1e-6 || std::hypot(x[0], x[1]) > 1e-5) {
        std::fprintf(stderr, "optimize stalled: grad=%g x=(%g,%g)\n", r.grad_norm, x[0],
                     x[1]);
        return 2;
    }
    xts::optimize::Solver solver(xts::optimize::Method::Lbfgs, xts::optimize::Control{}, 2);
    double lo[2] = {-1.0, -2.0};
    double hi[2] = {1.0, 2.0};
    int st = solver.set_box(lo, hi, 2);
    int st_ipm = solver.set_highs_solver(RGMIN_HIGHS_IPM);
    int st_x = solver.set_highs_crossover(RGMIN_HIGHS_CROSSOVER_OFF);
    std::printf("optimize nit=%zu grad=%g set_box=%d ipm=%d xover=%d\n", r.nit,
                r.grad_norm, st, st_ipm, st_x);
    return 0;
}
