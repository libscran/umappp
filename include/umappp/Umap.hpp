#ifndef UMAPPP_UMAP_HPP
#define UMAPPP_UMAP_HPP

#include "NeighborList.hpp"
#include "combine_neighbor_sets.hpp"
#include "find_ab.hpp"
#include "neighbor_similarities.hpp"
#include "optimize_layout.hpp"
#include "spectral_init.hpp"

namespace umappp {

class Umap {
private:
    double local_connectivity = 1.0;

    double bandwidth = 1;

    double mix_ratio = 1;

    double spread = 1;

    double min_dist = 0.01;

    double a = 0;

    double b = 0;

    double gamma = 1;

    bool init = true;

    int num_epochs = 500;

    double learning_rate = 1; 

    double negative_sample_rate = 5;

public:
    struct Status {
        Status(EpochData e, double a_, double b_) : epochs(std::move(e)), a(a_), b(b_) {}
        EpochData epochs; 
        double a;
        double b;
    };

    Status initialize(NeighborList x, int ndim, double* embedding) const {
        neighbor_similarities(x, local_connectivity, bandwidth);
        combine_neighbor_sets(x, mix_ratio);

        // Running spectral initialization.
        if (init) {
            spectral_init(x, ndim, embedding);
        }

        // Finding a good a/b pair.
        double a_ = a;
        double b_ = b;
        if (a_ <= 0 || b_ <= 0) {
            auto found = find_ab(spread, min_dist);
            a_ = found.first;
            b_ = found.second;
        }

        return Status(similarities_to_epochs(x, num_epochs), a_, b_);
    }

public:
    void run(Status& s, int ndim, double* embedding) const {
        optimize_layout(
            ndim,
            embedding,
            s.epochs,
            s.a,
            s.b,
            gamma,
            learning_rate,
            negative_sample_rate
        );
        return;
    }
};

}

#endif
