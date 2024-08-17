#ifndef UMAPPP_FIND_AB_HPP
#define UMAPPP_FIND_AB_HPP

#include <cmath>
#include <vector>

namespace umappp {

namespace internal {

/*
 * This function attempts to find 'a' and 'b' to fit:
 *
 * y ~ 1/(1 + a * x^(2 * b)) 
 *
 * against the curve:
 *
 * pmin(1, exp(-(x - d) / s))
 *
 * where 'd' is the min_dist and 's' is the spread.
 *
 * We do so by minimizing the least squares difference at grid points. The
 * original uwot:::find_ab_params does this via R's inbuilt nls() function, so
 * we follow its lead by using the Gauss-Newton method. (We add some
 * Levenbug-style dampening to guarantee convergence.)
 *
 * Derivatives were obtained using the following code in R:
 *
 * > delta <- expression((1/(1+ a * x^(2*b)) - y)^2)
 * > D(delta, "a")
 * > D(delta, "b")
 * > D(D(delta, "a"), "a")
 * > D(D(delta, "b"), "a")
 * > D(D(delta, "b"), "b")
 *
 * To test, evaluate with:
 *
 * > a <- 1.55461
 * > b <- 0.743147
 * > s <- 1
 * > m <- 0.05
 * > x <- seq(0, 3 * s, length.out=301)[-1]
 * > y <- pmin(1, exp(-(x - m)/s))
 * > sum(eval(delta))
 */
 
template<typename Float_>
std::pair<Float_, Float_> find_ab(Float_ spread, Float_ min_dist, Float_ grid = 300, Float_ limit = 0.5, int iter = 50, Float_ tol = 1e-6) {
    // Compute the x and y coordinates of the expected distance curve.
    std::vector<Float_> grid_x(grid), grid_y(grid), log_x(grid);
    const Float_ delta = spread * 3 / grid;
    for (int g = 0; g < grid; ++g) {
        grid_x[g] = (g + 1) * delta; // +1 to avoid meaningless least squares result at x = 0, where both curves have y = 1 (and also the derivative w.r.t. b is not defined).
        log_x[g] = std::log(grid_x[g]);
        grid_y[g] = (grid_x[g] <= min_dist ? 1 : std::exp(- (grid_x[g] - min_dist) / spread));
    }

    // Starting estimates, obtained by solving for 'exp(- (x - d) / s) = limit'.
    // We use 'limit = 0.5' because that's where most interesting stuff happens in this curve.
    Float_ x_half = std::log(limit) * -spread + min_dist;
    Float_ d_half = limit / -spread;
    Float_ b = - d_half * x_half / (1 / limit - 1) / (2 * limit * limit);
    Float_ a = (1 / limit - 1) / std::pow(x_half, 2 * b);

    std::vector<Float_> observed_y(grid), xpow(grid);
    auto compute_ss = [&](Float_ A, Float_ B) -> Float_ {
        for (int g = 0; g < grid; ++g) {
            xpow[g] = std::pow(grid_x[g], 2 * B);
            observed_y[g] = 1 / (1 + A * xpow[g]);
        }

        Float_ ss = 0;
        for (int g = 0; g < grid; ++g) {
            ss += (grid_y[g] - observed_y[g]) * (grid_y[g] - observed_y[g]);
        }

        return ss;
    };
    Float_ ss = compute_ss(a, b);

    for (int it = 0; it < iter; ++it) {
        // Computing the first and second derivatives of the sum of squared differences.
        Float_ da = 0, db = 0, daa = 0, dab = 0, dbb = 0;
        for (int g = 0; g < grid; ++g) {
            const Float_& gy = grid_y[g];
            const Float_& oy = observed_y[g];

            const Float_& x2b = xpow[g];
            const Float_ logx2 = log_x[g] * 2;
            const Float_ delta = oy - gy;

            // -(2 * (x^(2 * b)/(1 + a * x^(2 * b))^2 * (1/(1 + a * x^(2 * b)) - y)))
            da += -2 * x2b * oy * oy * delta;

            // -(2 * (a * (x^(2 * b) * (log(x) * 2))/(1 + a * x^(2 * b))^2 * (1/(1 + a * x^(2 * b)) - y)))
            db += -2 * a * x2b * logx2 * oy * oy * delta;

            // 2 * (
            //     x^(2 * b)/(1 + a * x^(2 * b))^2 * (x^(2 * b)/(1 + a * x^(2 * b))^2) 
            //     + x^(2 * b) * (2 * (x^(2 * b) * (1 + a * x^(2 * b))))/((1 + a * x^(2 * b))^2)^2 * (1/(1 + a * x^(2 * b)) - y)
            // ) 
            daa += 2 * (
                x2b * oy * oy * x2b * oy * oy
                + x2b * 2 * x2b * oy * oy * oy * delta
            );

            //-(2 * 
            //    (
            //        (
            //            (x^(2 * b) * (log(x) * 2))/(1 + a * x^(2 * b))^2 
            //            - a * (x^(2 * b) * (log(x) * 2)) * (2 * (x^(2 * b) * (1 + a * x^(2 * b))))/((1 + a * x^(2 * b))^2)^2
            //        ) 
            //        * (1/(1 + a * x^(2 * b)) - y) 
            //        - a * (x^(2 * b) * (log(x) * 2))/(1 + a * x^(2 * b))^2 * (x^(2 * b)/(1 + a * x^(2 * b))^2)
            //    )
            //)
            dab += -2 * (
                (
                    x2b * logx2 * oy * oy
                    - a * x2b * logx2 * 2 * x2b * oy * oy * oy
                ) * delta
                - a * x2b * logx2 * oy * oy * x2b * oy * oy
            );

            // -(2 * 
            //     (
            //         (
            //             a * (x^(2 * b) * (log(x) * 2) * (log(x) * 2))/(1 + a * x^(2 * b))^2 
            //             - a * (x^(2 * b) * (log(x) * 2)) * (2 * (a * (x^(2 * b) * (log(x) * 2)) * (1 + a * x^(2 * b))))/((1 + a * x^(2 * b))^2)^2
            //         ) 
            //         * (1/(1 + a * x^(2 * b)) - y) 
            //         - a * (x^(2 * b) * (log(x) * 2))/(1 + a * x^(2 * b))^2 * (a * (x^(2 * b) * (log(x) * 2))/(1 + a * x^(2 * b))^2)
            //     )
            // ) 
            dbb += -2 * (
                (
                    (a * x2b * logx2 * logx2 * oy * oy)
                    - (a * x2b * logx2 * 2 * a * x2b * logx2 * oy * oy * oy)
                ) * delta 
                - a * x2b * logx2 * oy * oy * a * x2b * logx2 * oy * oy
            );
        }

        // Applying the Newton iterations with damping.
        Float_ determinant = daa * dbb - dab * dab;
        const Float_ delta_a = (da * dbb - dab * db) / determinant;
        const Float_ delta_b = (- da * dab + daa * db) / determinant; 

        Float_ ss_next = 0;
        Float_ factor = 1;
        for (int inner = 0; inner < 10; ++inner, factor /= 2) {
            ss_next = compute_ss(a - factor * delta_a, b - factor * delta_b);
            if (ss_next < ss) {
                break;
            }
        }

        if (ss && 1 - ss_next/ss > tol) {
            a -= factor * delta_a;
            b -= factor * delta_b;
            ss = ss_next;
        } else {
            break;
        }
    }

    return std::make_pair(a, b);
}

}

}

#endif
