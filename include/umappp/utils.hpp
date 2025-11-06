#ifndef UMAPPP_UTILS_HPP
#define UMAPPP_UTILS_HPP

#include <type_traits>

namespace umappp {

template<typename Input_>
using I = typename std::remove_cv<typename std::remove_reference<Input_>::type>::type;

}

#endif
