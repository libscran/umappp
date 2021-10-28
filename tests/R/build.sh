# This pulls in the same external libraries and makes them available to Rcpp's
# compilation. We don't use RcppEigen because the version there is too old to
# contain the features used by irlba.

set -e
set -u

ROOT=../../build/_deps

rm -f irlba
ln -s ${ROOT}/irlba-src/include/irlba .

rm -f Eigen
ln -s ${ROOT}/eigen3-src/Eigen .

rm -f aarand
ln -s ${ROOT}/aarand-src/include/aarand .

rm -f umappp
ln -s ../../include/umappp .
