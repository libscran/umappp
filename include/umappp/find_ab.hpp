#ifndef UMAPPP_FIND_AB_HPP
#define UMAPPP_FIND_AB_HPP

#include <cmath>
#include <vector>

#include "sanisizer/sanisizer.hpp"

namespace umappp {

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
 * Levenberg-Marquadt-style dampening to guarantee convergence.)
 *
 * Derivatives were obtained using the following code in R:
 *
 * > delta <- expression(y - 1/(1+ a * x^(2*b)))
 * > D(delta, "a")
 * > D(delta, "b")
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
std::pair<Float_, Float_> find_ab(const Float_ spread, const Float_ min_dist) {
    constexpr std::size_t grid = 300;
    auto grid_x = sanisizer::create<std::vector<Float_> >(grid);
    auto grid_y = sanisizer::create<std::vector<Float_> >(grid);
    auto log_x = sanisizer::create<std::vector<Float_> >(grid);

    // Compute the x and y coordinates of the expected distance curve.
    const Float_ delta = spread * 3 / grid;
    for (std::size_t g = 0; g < grid; ++g) {
        grid_x[g] = (g + 1) * delta; // +1 to avoid meaningless least squares result at x = 0, where both curves have y = 1 (and also the derivative w.r.t. b is not defined).
        log_x[g] = std::log(grid_x[g]);
        grid_y[g] = (grid_x[g] <= min_dist ? 1 : std::exp(- (grid_x[g] - min_dist) / spread));
    }

    // Starting estimates, obtained by matching the coordinates/gradients of
    // the two curves (ignoring the pmin) where 'exp(- (x - d) / s) = limit'.
    // We use 'limit = 0.5' because that's where most interesting stuff
    // happens, given that the curve is bounded between 0 and 1 on the y-axis.
    constexpr Float_ limit = 0.5;
    const Float_ x_half = std::log(limit) * -spread + min_dist; // guaranteed > 0, as log(limit) is negative.
    const Float_ d_half = limit / -spread; // first derivative at x_half.
    Float_ b = - d_half * x_half / (1 / limit - 1) / (2 * limit * limit);
    Float_ a = (1 / limit - 1) / std::pow(x_half, 2 * b);

    auto fit_y = sanisizer::create<std::vector<Float_> >(grid);
    auto xpow = sanisizer::create<std::vector<Float_> >(grid);
    auto grid_resid = sanisizer::create<std::vector<Float_> >(grid);

    auto compute_ss = [&](const Float_ A, const Float_ B) -> Float_ {
        Float_ ss = 0;
        for (std::size_t g = 0; g < grid; ++g) {
            xpow[g] = std::pow(grid_x[g], 2 * B);
            fit_y[g] = 1 / (1 + A * xpow[g]);
            grid_resid[g] = grid_y[g] - fit_y[g];
            ss += grid_resid[g] * grid_resid[g];
        }
        return ss;
    };
    Float_ ss = compute_ss(a, b);

    // Starting with basically no Levenberg-Marquardt dampening,
    // under the assumption that the starting estimates are pretty good.
    Float_ lm_dampener = 0;

    constexpr int gn_iter = 50; // i.e., Gauss-Newton iterations.
    for (int it = 0; it < gn_iter; ++it) {
        /* Using Wikipedia's notation (https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm):
         *
         * J^T J = [ da2   dadb ]
         *         [ dadb  db2  ] 
         *
         * J^T r(Beta) = [ da_resid ]
         *               [ db_resid ]
         */
        Float_ da2 = 0, db2 = 0, dadb = 0, da_resid = 0, db_resid = 0;

        for (std::size_t g = 0; g < grid; ++g) {
            const Float_ x2b = xpow[g]; // set by the last compute_ss() call.
            const Float_ oy = fit_y[g]; // ditto
            const Float_ resid = grid_resid[g]; // ditto

            // x^(2 * b)/(1 + a * x^(2 * b))^2
            const Float_ da = x2b * oy * oy;

            // a * (x^(2 * b) * (log(x) * 2))/(1 + a * x^(2 * b))^2
            const Float_ db = a * (log_x[g] * 2) * da; // reusing expression above.

            da2 += da * da;
            db2 += db * db;
            dadb += da * db;
            da_resid += da * resid;
            db_resid += db * resid;
        }

        bool okay = false;
        Float_ candidate_a, candidate_b, ss_next;

        // To get from epsilon to max_dampener for a double-precision Float_ is
        // 62 Levenberg-Marquardt iterations; that should be acceptable for the
        // pathological case, as it is comparable to gn_iter.
        constexpr Float_ max_dampener = 1024; 

        while (lm_dampener < max_dampener) {
            const Float_ mult = 1 + lm_dampener; 
            const Float_ damped_da2 = da2 * mult;
            const Float_ damped_db2 = db2 * mult;

            const Float_ determinant = damped_da2 * damped_db2 - dadb * dadb;
            const Float_ delta_a = - (da_resid * damped_db2 - dadb * db_resid) / determinant;
            const Float_ delta_b = - (- da_resid * dadb + damped_da2 * db_resid) / determinant;

            candidate_a = a + delta_a;
            candidate_b = b + delta_b;

            ss_next = compute_ss(candidate_a, candidate_b);
            if (ss_next < ss) {
                okay = true;
                lm_dampener /= 2;
                break;
            }

            if (lm_dampener == 0) {
                lm_dampener = std::numeric_limits<Float_>::epsilon();
            } else {
                lm_dampener *= 2;
            }
        }

        if (!okay) { // Give up, I guess... hopefully this doesn't cause too much damage.
            break;
        }

        constexpr Float_ tol = 1e-6;
        if (ss - ss_next <= ss * tol) { // Converged successfully within the relative tolerance.
            break;
        }

        a = candidate_a;
        b = candidate_b;
        ss = ss_next;
    }

    return std::make_pair(a, b);
}

}

#endif
