#ifndef UMAPPP_UTILS_HPP
#define UMAPPP_UTILS_HPP

#include <limits>
#include <type_traits>

namespace umappp {

template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(const Input_ x) {
    return x;
}

}

#endif
