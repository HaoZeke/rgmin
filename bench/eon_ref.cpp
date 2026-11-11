// Copyright (c) rgmin developers
//
// Numerical core of the eOn client optimizers, transcribed from
// client/LBFGS.cpp, client/FIRE.cpp and client/ConjugateGradients.cpp
// at 579738a9387e8d0ea55496f45dd057812cfab892
// (https://github.com/TheochemUI/eOn). Defaults are
// include/eon/generated/ParametersSSOTDefaults.h at that commit:
// L-BFGS memory 20, inverse curvature 0.01, auto scale, distance reset,
// angle reset, curvature "reset", finite difference 0.01, max move 0.2,
// accept "none"; FIRE time step 1, max time step 2.5, Nmin 5, alpha 0.1;
// CG line search off.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kMaxMove = 0.2;
constexpr double kFd = 0.01;
constexpr double kInv = 0.01;
constexpr double kEps = 1e-30;
constexpr int kBudget = 20000;
constexpr int kRepeats = 5;

struct Fg {
    virtual ~Fg() = default;
    virtual void eval(const std::vector<double> &x, double &e, std::vector<double> &g) = 0;
};

double dot(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

double norm(const std::vector<double> &a) { return std::sqrt(dot(a, a)); }

double max_atom(const std::vector<double> &v) {
    double m = 0.0;
    const size_t n = v.size();
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        const double r = std::sqrt(v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2]);
        m = std::max(m, r);
    }
    if (i < n) {
        double r2 = 0.0;
        for (; i < n; ++i) {
            r2 += v[i] * v[i];
        }
        m = std::max(m, std::sqrt(r2));
    }
    return m;
}

std::vector<double> cap_atom(std::vector<double> v, double cap) {
    const double m = max_atom(v);
    if (m > cap && m > 0.0) {
        const double s = cap / m;
        for (double &x : v) {
            x *= s;
        }
    }
    return v;
}

std::vector<double> scale(const std::vector<double> &v, double s) {
    std::vector<double> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        o[i] = s * v[i];
    }
    return o;
}

std::vector<double> add(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> o(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        o[i] = a[i] + b[i];
    }
    return o;
}

std::vector<double> sub(const std::vector<double> &a, const std::vector<double> &b) {
    return add(a, scale(b, -1.0));
}

double stop_grad(const std::vector<double> &g, bool cluster) {
    if (!cluster) {
        double m = 0.0;
        for (double v : g) {
            m = std::max(m, std::abs(v));
        }
        return m;
    }
    return max_atom(g);
}

struct Counters {
    int evals = 0;
};

void call(Fg &fg, Counters &c, const std::vector<double> &x, double &e, std::vector<double> &g) {
    fg.eval(x, e, g);
    c.evals += 1;
}

struct Lbfgs {
    int memory = 20;
    std::deque<std::vector<double>> s, y;
    std::deque<double> rho;
    std::vector<double> r_prev, f_prev;
    int iteration = 0;

    void reset() {
        s.clear();
        y.clear();
        rho.clear();
    }

    std::vector<double> get_step(Fg &fg, Counters &c, const std::vector<double> &r,
                                 const std::vector<double> &force, double max_move) {
        double h0 = kInv;
        if (iteration > 0) {
            const std::vector<double> dr = sub(r, r_prev);
            const std::vector<double> yv = sub(f_prev, force);
            const double sy = dot(dr, yv);
            const double yy = dot(yv, yv);
            const double curv = (sy != 0.0 && std::isfinite(sy) && std::isfinite(yy)) ? yy / sy : -1.0;
            if (curv < 0.0) {
                reset();
                return cap_atom(scale(force, 1000.0), max_move);
            }
            if (sy > 0.0 && yy > 0.0) {
                h0 = sy / yy;
            }
        } else {
            const double fn = norm(force);
            if (fn > 0.0) {
                const std::vector<double> u = scale(force, 1.0 / fn);
                const std::vector<double> probe = add(r, scale(u, kFd));
                double ep = 0.0;
                std::vector<double> gp(r.size());
                call(fg, c, probe, ep, gp);
                std::vector<double> dg = gp;
                for (size_t i = 0; i < dg.size(); ++i) {
                    dg[i] += force[i];
                }
                const double curv = dot(dg, u) / kFd;
                if (!(curv > 0.0) || !std::isfinite(curv)) {
                    reset();
                    return cap_atom(scale(force, 1000.0), max_move);
                }
                h0 = 1.0 / curv;
            }
        }

        std::vector<double> q(force.size());
        for (size_t i = 0; i < force.size(); ++i) {
            q[i] = -force[i];
        }
        const int n = static_cast<int>(s.size());
        std::vector<double> alpha(static_cast<size_t>(n));
        for (int k = n - 1; k >= 0; --k) {
            alpha[static_cast<size_t>(k)] = rho[static_cast<size_t>(k)] * dot(s[static_cast<size_t>(k)], q);
            const double a = alpha[static_cast<size_t>(k)];
            for (size_t i = 0; i < q.size(); ++i) {
                q[i] -= a * y[static_cast<size_t>(k)][i];
            }
        }
        for (double &v : q) {
            v *= h0;
        }
        for (int k = 0; k < n; ++k) {
            const double b = rho[static_cast<size_t>(k)] * dot(y[static_cast<size_t>(k)], q);
            const double coef = alpha[static_cast<size_t>(k)] - b;
            for (size_t i = 0; i < q.size(); ++i) {
                q[i] += coef * s[static_cast<size_t>(k)][i];
            }
        }
        std::vector<double> d(q.size());
        for (size_t i = 0; i < q.size(); ++i) {
            d[i] = -q[i];
        }
        if (max_atom(d) >= max_move) {
            reset();
            return cap_atom(scale(force, h0), max_move);
        }
        const double dn = norm(d);
        const double fn = norm(force);
        if (dn > 0.0 && fn > 0.0 && dot(d, force) < 0.0) {
            reset();
            return cap_atom(scale(force, h0), max_move);
        }
        return cap_atom(d, max_move);
    }

    void update(const std::vector<double> &r1, const std::vector<double> &r0,
                const std::vector<double> &f1, const std::vector<double> &f0) {
        std::vector<double> s0 = sub(r1, r0);
        std::vector<double> y0 = sub(f0, f1);
        const double sy = dot(s0, y0);
        if (!std::isfinite(sy) || std::abs(sy) < kEps) {
            reset();
            return;
        }
        rho.push_back(1.0 / sy);
        s.push_back(std::move(s0));
        y.push_back(std::move(y0));
        while (static_cast<int>(s.size()) > memory) {
            s.pop_front();
            y.pop_front();
            rho.pop_front();
        }
    }

    void step(Fg &fg, Counters &c, std::vector<double> &x, double & /*e*/, std::vector<double> &g,
              double max_move) {
        std::vector<double> force(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            force[i] = -g[i];
        }
        if (iteration > 0) {
            update(x, r_prev, force, f_prev);
        }
        const std::vector<double> dr = get_step(fg, c, x, force, max_move);
        r_prev = x;
        f_prev = force;
        x = add(x, dr);
        double e = 0.0;
        call(fg, c, x, e, g);
        iteration += 1;
    }
};

struct Fire {
    double dt = 1.0;
    double dt_max = 2.5;
    double alpha = 0.1;
    double alpha0 = 0.1;
    int nmin = 5;
    int n = 0;
    std::vector<double> vel;
    bool stalled = false;

    void step(Fg &fg, Counters &c, std::vector<double> &x, std::vector<double> &g, double max_move) {
        if (vel.empty()) {
            vel.assign(x.size(), 0.0);
        }
        std::vector<double> force(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            force[i] = -g[i];
            vel[i] += force[i] * dt;
        }
        std::vector<double> dx = cap_atom(scale(vel, dt), max_move);
        x = add(x, dx);
        double e = 0.0;
        call(fg, c, x, e, g);
        for (size_t i = 0; i < g.size(); ++i) {
            force[i] = -g[i];
        }
        const double fn = norm(force);
        const double vn = norm(vel);
        const double p = dot(force, vel);
        if (fn > 0.0) {
            const double scale_v = alpha * vn / fn;
            for (size_t i = 0; i < vel.size(); ++i) {
                vel[i] = (1.0 - alpha) * vel[i] + scale_v * force[i];
            }
        }
        if (p >= 0.0) {
            n += 1;
            if (n > nmin) {
                dt = std::min(dt * 1.1, dt_max);
                alpha *= 0.99;
            }
        } else {
            dt *= 0.5;
            std::fill(vel.begin(), vel.end(), 0.0);
            alpha = alpha0;
            n = 0;
        }
        if (dt < 1e-6) {
            stalled = true;
        }
    }
};

struct Cg {
    std::vector<double> direction, direction_old, force, force_old;
    int cg_i = 0;

    void direction_of() {
        const double b = dot(force_old, force_old);
        const double a = std::abs(dot(force, force_old));
        double gamma = 0.0;
        if (b > 0.0 && a < 0.5 * b) {
            gamma = dot(force, sub(force, force_old)) / b;
        }
        if (direction.empty()) {
            direction = force;
        } else {
            direction = add(force, scale(direction_old, gamma));
        }
        direction_old = direction;
        force_old = force;
        cg_i += 1;
    }

    void step(Fg &fg, Counters &c, std::vector<double> &x, std::vector<double> &g, double max_move) {
        force.resize(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            force[i] = -g[i];
        }
        if (force_old.empty()) {
            force_old.assign(g.size(), 0.0);
            direction_old.assign(g.size(), 0.0);
        }
        direction_of();
        const double dn = norm(direction);
        std::vector<double> un = dn > 0.0 ? scale(direction, 1.0 / dn) : direction;
        const std::vector<double> probe = add(x, scale(un, kFd));
        double ep = 0.0;
        std::vector<double> gp(g.size());
        call(fg, c, probe, ep, gp);
        std::vector<double> force_after(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            force_after[i] = -gp[i];
        }
        const double p1 = dot(force, un);
        const double p2 = dot(force_after, un);
        const double curv = (p1 - p2) / kFd;
        double step_size = max_move;
        if (curv > 0.0 && std::isfinite(curv)) {
            step_size = p1 / curv;
        }
        if (std::abs(step_size) > max_move) {
            step_size = std::copysign(max_move, step_size);
        }
        x = add(x, cap_atom(scale(un, step_size), max_move));
        double e = 0.0;
        call(fg, c, x, e, g);
        if (step_size >= max_move) {
            std::fill(direction_old.begin(), direction_old.end(), 0.0);
            std::fill(force_old.begin(), force_old.end(), 0.0);
        }
    }
};

struct Rosen : Fg {
    void eval(const std::vector<double> &x, double &e, std::vector<double> &g) override {
        e = 0.0;
        g.assign(x.size(), 0.0);
        for (size_t i = 0; i + 1 < x.size(); ++i) {
            const double a = 1.0 - x[i];
            const double b = x[i + 1] - x[i] * x[i];
            e += a * a + 100.0 * b * b;
            g[i] += -2.0 * a - 400.0 * x[i] * b;
            g[i + 1] += 200.0 * b;
        }
    }
};

struct Lj : Fg {
    void eval(const std::vector<double> &x, double &e, std::vector<double> &g) override {
        const int n = static_cast<int>(x.size() / 3);
        e = 0.0;
        g.assign(x.size(), 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double d[3], r2 = 0.0;
                for (int a = 0; a < 3; ++a) {
                    d[a] = x[3 * i + a] - x[3 * j + a];
                    r2 += d[a] * d[a];
                }
                const double s6 = std::pow(r2, -3.0);
                e += 4.0 * (s6 * s6 - s6);
                const double du = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
                for (int a = 0; a < 3; ++a) {
                    g[3 * i + a] += du * d[a];
                    g[3 * j + a] -= du * d[a];
                }
            }
        }
    }
};

struct Morse : Fg {
    double cell[3];
    explicit Morse(const double c[3]) {
        cell[0] = c[0];
        cell[1] = c[1];
        cell[2] = c[2];
    }
    void eval(const std::vector<double> &x, double &e, std::vector<double> &g) override {
        const int n = static_cast<int>(x.size() / 3);
        e = 0.0;
        g.assign(x.size(), 0.0);
        const double d0 = 0.7102, aa = 1.6047, r0 = 2.897;
        const double xc = std::exp(-aa * (7.0 - r0));
        const double shift = d0 * (xc * xc - 2.0 * xc);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double d[3];
                for (int a = 0; a < 3; ++a) {
                    d[a] = x[3 * i + a] - x[3 * j + a];
                    d[a] -= cell[a] * std::round(d[a] / cell[a]);
                }
                const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                if (r >= 7.0) {
                    continue;
                }
                const double ex = std::exp(-aa * (r - r0));
                e += d0 * (ex * ex - 2.0 * ex) - shift;
                const double du = 2.0 * d0 * aa * ex * (1.0 - ex);
                for (int a = 0; a < 3; ++a) {
                    g[3 * i + a] += du * d[a] / r;
                    g[3 * j + a] -= du * d[a] / r;
                }
            }
        }
    }
};

std::vector<double> rosen_start(int n) {
    std::vector<double> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = (i % 2 == 0) ? -1.2 : 1.0;
    }
    return x;
}

std::vector<double> grid_start(int atoms) {
    const int side = static_cast<int>(std::ceil(std::cbrt(static_cast<double>(atoms))));
    std::vector<double> x;
    x.reserve(static_cast<size_t>(3 * atoms));
    for (int k = 0; k < atoms; ++k) {
        const int cell[3] = {k % side, (k / side) % side, k / (side * side)};
        for (int a = 0; a < 3; ++a) {
            x.push_back(1.1 * cell[a] + 0.03 * std::sin(3.0 * k + a));
        }
    }
    return x;
}

bool read_con(const std::string &path, std::vector<double> &x, double cell[3]) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (n == 2) {
            std::istringstream ss(line);
            ss >> cell[0] >> cell[1] >> cell[2];
        }
        if (line.rfind("Coordinates", 0) == 0) {
            break;
        }
        n += 1;
    }
    x.clear();
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        double a, b, c;
        if (!(ss >> a >> b >> c)) {
            break;
        }
        x.push_back(a);
        x.push_back(b);
        x.push_back(c);
    }
    return !x.empty();
}

struct Result {
    int evals = 0;
    double ms = 0.0;
    double energy = 0.0;
    double grad = 0.0;
    bool converged = false;
};

template <class Run>
Result time_runs(Run run) {
    std::vector<double> times;
    Result last;
    for (int rep = 0; rep < kRepeats; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        last = run();
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    last.ms = times[times.size() / 2];
    return last;
}

Result solve(const std::string &arm, Fg &problem, std::vector<double> x0, bool cluster, double tol) {
    return time_runs([&]() {
        Counters c;
        std::vector<double> x = x0;
        double e = 0.0;
        std::vector<double> g(x.size());
        call(problem, c, x, e, g);
        bool converged = false;
        if (arm == "lbfgs") {
            Lbfgs opt;
            for (int it = 0; it < kBudget; ++it) {
                if (stop_grad(g, cluster) < tol) {
                    converged = true;
                    break;
                }
                opt.step(problem, c, x, e, g, kMaxMove);
            }
        } else if (arm == "fire") {
            Fire opt;
            for (int it = 0; it < kBudget && !opt.stalled; ++it) {
                if (stop_grad(g, cluster) < tol) {
                    converged = true;
                    break;
                }
                opt.step(problem, c, x, g, kMaxMove);
            }
        } else {
            Cg opt;
            for (int it = 0; it < kBudget; ++it) {
                if (stop_grad(g, cluster) < tol) {
                    converged = true;
                    break;
                }
                opt.step(problem, c, x, g, kMaxMove);
            }
        }
        double ef = 0.0;
        std::vector<double> gf(x.size());
        problem.eval(x, ef, gf);
        Result r;
        r.evals = c.evals;
        r.energy = ef;
        r.grad = stop_grad(gf, cluster);
        r.converged = converged || r.grad < tol;
        return r;
    });
}

void quad_check() {
    struct Quad : Fg {
        void eval(const std::vector<double> &x, double &e, std::vector<double> &g) override {
            e = 0.5 * dot(x, x);
            g = x;
        }
    } q;
    Counters c;
    std::vector<double> x = {1.0, 0.0, 0.0};
    double e = 0.0;
    std::vector<double> g;
    call(q, c, x, e, g);
    Lbfgs opt;
    opt.step(q, c, x, e, g, kMaxMove);
    if (c.evals != 3 || std::abs(x[0] - 0.8) > 1e-12) {
        std::cerr << "eOn L-BFGS quadratic check failed, evals " << c.evals << " x0 " << x[0] << "\n";
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char **argv) {
    std::string root = argc > 1 ? argv[1] : ".";
    std::string out_path = argc > 2 ? argv[2] : "bench/raw/eon.csv";
    quad_check();
    std::ofstream out(out_path);
    out << "problem,arm,library,evals,energy,grad,median_ms,converged\n";
    auto write = [&](const std::string &problem, const std::string &arm, const Result &r) {
        out << problem << ",eon-" << arm << ",eon," << r.evals << "," << std::scientific << r.energy
            << "," << r.grad << "," << std::fixed << r.ms << "," << (r.converged ? "yes" : "no") << "\n";
        out.flush();
        std::cout << problem << ",eon-" << arm << ",eon," << r.evals << "," << r.energy << "," << r.grad
                  << "," << r.ms << "," << (r.converged ? "yes" : "no") << "\n";
    };
    const std::string arms[] = {"lbfgs", "fire", "cg"};
    {
        Rosen rose;
        for (const auto &arm : arms) {
            write("Rosenbrock2", arm, solve(arm, rose, rosen_start(2), false, 1e-5));
            write("Rosenbrock10", arm, solve(arm, rose, rosen_start(10), false, 1e-5));
        }
    }
    {
        Lj lj;
        for (int atoms : {7, 13, 38}) {
            for (const auto &arm : arms) {
                write("LJ" + std::to_string(atoms), arm, solve(arm, lj, grid_start(atoms), true, 0.01));
            }
        }
    }
    {
        double cell[3] = {};
        std::vector<double> x;
        if (!read_con(root + "/tests/data/morse_pos_0000.con", x, cell)) {
            std::cerr << "missing morse fixture\n";
            return 1;
        }
        Morse morse(cell);
        for (const auto &arm : arms) {
            write("MorsePt256", arm, solve(arm, morse, x, true, 0.01));
        }
    }
    return 0;
}
