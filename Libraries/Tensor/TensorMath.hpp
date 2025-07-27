#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "Tensor.hpp"

namespace swnumeric {

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///
/// NORMS todo: add blas specializations
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

template <typename T, size_t sz>
T norm2(const Vector<sz, T>& x) {
	T sumsq = T(0.0);
	for (size_t i = 0; i < x.size(); i++) {
		sumsq += x[i] * x[i];
	}
	return std::sqrt(sumsq);
}

template <typename T, size_t sz>
T norm2Sq(const Vector<sz, T>& x) {
	T sumsq = T(0.0);
	for (size_t i = 0; i < x.size(); i++) {
		sumsq += x[i] * x[i];
	}
	return (sumsq);
}

template <typename T, size_t sz>
T normInf(const Vector<sz, T>& x) {
	T mmax = T(0.0);
	for (size_t i = 0; i < x.size(); i++) {
		mmax = std::max(mmax, std::abs(x[i]));
	}
	return mmax;
}

template <typename T, size_t sz>
T normNegInf(const Vector<sz, T>& x) {
	T mmin = std::numeric_limits<T>::max();
	for (size_t i = 0; i < x.size(); i++) {
		mmin = std::min(mmin, std::abs(x[i]));
	}
	return mmin;
}

template <typename T, size_t sz>
T norm1(const Vector<sz, T>& x) {
	T sum = T(0.0);
	for (size_t i = 0; i < x.size(); i++) {
		sum += std::abs(x[i]);
	}
	return sum;
}

template <typename T, size_t sz>
T norm0(const Vector<sz, T>& x) {
	uint64_t nonZero = 0;
	for (size_t i = 0; i < x.size(); i++) {
		if (x[i] != 0.0) nonZero++;
	}
	return T(nonZero);
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
///
///       SIMPLE VECTOR OPERATIONS
///
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

template <typename T, size_t sz>
void normalizeInplace(Vector<sz, T>& x) {
	T nrm = norm2(x);
	x /= nrm;
}

template <typename T, size_t sz>
Vector<sz, T> normalize(const Vector<sz, T>& x) {
	Vector<sz, T> y = x;
	T nrm = norm2(x);
	y /= nrm;
	return y;
}

inline Vector3 cross(const Vector3& v1, const Vector3& v2) {
	Vector3 x;
	x[0] = v1[1] * v2[2] - v1[2] * v2[1];
	x[1] = v1[2] * v2[0] - v1[0] * v2[2];
	x[2] = v1[0] * v2[1] - v1[1] * v2[0];
	return x;
}

inline void crossTo(Vector3& x, const Vector3& v1, const Vector3& v2) {
	x[0] = v1[1] * v2[2] - v1[2] * v2[1];
	x[1] = v1[2] * v2[0] - v1[0] * v2[2];
	x[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

inline int crossSign(const Vector2& v1, const Vector2& v2) {
	return v2[0] * v1[1] - v2[1] * v1[0] > 0.0 ? 1 : -1;
}

template <typename T, size_t sz1, size_t sz2>
T dot(const Vector<sz1, T>& x1, const Vector<sz2>& x2) {
	T sum = 0.0;
	for (size_t i = 0; i < x1.size(); i++) {
		sum += *x1(i) * *x2(i);
	}
	return sum;
}

}  // namespace swnumeric
