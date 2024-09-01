#ifndef CUSTOM_PARALLEL_H
#define CUSTOM_PARALLEL_H

#include <vector>
#include <thread>

template<class Function_>
void default_parallelize(size_t nthreads, size_t n, Function_ f) {
    size_t jobs_per_worker = (n / nthreads) + (n % nthreads > 0);
    size_t start = 0;
    std::vector<std::thread> jobs;
    jobs.reserve(nthreads);
    
    for (size_t w = 0; w < nthreads && start < n; ++w) {
        size_t len = std::min(n - start, jobs_per_worker);
        jobs.emplace_back(f, w, start, len);
        start += len;
    }

    for (auto& job : jobs) {
        job.join();
    }
}

#define UMAPPP_CUSTOM_PARALLEL default_parallelize
#endif
