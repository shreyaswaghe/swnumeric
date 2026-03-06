#pragma once

#include <mkl_cblas.h>
#include <mkl_cblas_64.h>
#include <sys/cdefs.h>

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include "library/expression/Expression.h"

template <size_t N, std::floating_point T>
struct StaticVector
// : Expr<StaticVector<N, T>, T>
{
  static_assert(N > 0, "Length to StaticVector must be positive.");

  //
  using value_type = T;
  constexpr static size_t ctime_size = N;
  //
  T _data[N];

  // size
  constexpr size_t size() const noexcept { return N; }
  constexpr bool is_alloc() const noexcept { return true; }
  constexpr bool alloc(size_t /*N*/) const noexcept { return true; }

  // data access
  constexpr T* data() noexcept { return _data; }
  constexpr const T* data() const noexcept { return _data; }
  constexpr T& operator[](size_t i) noexcept {
    assert(i < N);
    return _data[i];
  }
  constexpr const T& operator[](size_t i) const noexcept {
    assert(i < N);
    return _data[i];
  }

  // iterator access
  constexpr const T* begin() const noexcept { return _data; }
  constexpr const T* end() const noexcept { return _data + N; }
  constexpr T* begin() noexcept { return _data; }
  constexpr T* end() noexcept { return _data + N; }

  // operator=
  constexpr StaticVector& operator=(const StaticVector& src) = default;

  template <typename ExprType>
  constexpr StaticVector& operator=(const Expr<ExprType, T>& src) {
    for (size_t i = 0; i < size(); i++) {
      data()[i] = src[i];
    }
  }
};

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator+(const StaticVector<N, T>& a,
                                             const StaticVector<N, T>& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a[i] + b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator-(const StaticVector<N, T>& a,
                                             const StaticVector<N, T>& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a[i] - b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator*(const StaticVector<N, T>& a,
                                             const StaticVector<N, T>& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a[i] * b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator/(const StaticVector<N, T>& a,
                                             const StaticVector<N, T>& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a[i] / b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator+(const StaticVector<N, T>& a,
                                             const T& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a[i] + b;
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator-(const StaticVector<N, T>& a,
                                             const T& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a[i] - b;
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator*(const StaticVector<N, T>& a,
                                             const T& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a[i] * b;
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator/(const StaticVector<N, T>& a,
                                             const T& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a[i] / b;
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator+(const T& a,
                                             const StaticVector<N, T>& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a + b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator-(const T& a,
                                             const StaticVector<N, T>& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a - b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator*(const T& a,
                                             const StaticVector<N, T>& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a * b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator/(const T& a,
                                             const StaticVector<N, T>& b) {
  StaticVector<N, T> c;
  for (size_t i = 0; i < N; i++) c[i] = a / b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T>& operator+=(StaticVector<N, T>& a,
                                               const StaticVector<N, T>& b) {
  for (size_t i = 0; i < N; i++) a[i] += b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator-=(StaticVector<N, T>& a,
                                              const StaticVector<N, T>& b) {
  for (size_t i = 0; i < N; i++) a[i] -= b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator*=(StaticVector<N, T>& a,
                                              const StaticVector<N, T>& b) {
  for (size_t i = 0; i < N; i++) a[i] *= b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator/=(StaticVector<N, T>& a,
                                              const StaticVector<N, T>& b) {
  for (size_t i = 0; i < N; i++) a[i] /= b[i];
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T>& operator+=(StaticVector<N, T>& a,
                                               const T& b) {
  for (size_t i = 0; i < N; i++) a[i] += b;
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator-=(StaticVector<N, T>& a,
                                              const T& b) {
  for (size_t i = 0; i < N; i++) a[i] -= b;
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator*=(StaticVector<N, T>& a,
                                              const T& b) {
  for (size_t i = 0; i < N; i++) a[i] *= b;
}

template <size_t N, std::floating_point T>
__always_inline StaticVector<N, T> operator/=(StaticVector<N, T>& a,
                                              const T& b) {
  for (size_t i = 0; i < N; i++) a[i] /= b;
}

template <std::floating_point T>
struct Vector : Expr<Vector<T>, T> {
  //
  using value_type = T;
  constexpr static size_t ctime_size = 0;

  //
  std::vector<T> _data;

  // constructor
  explicit Vector(size_t N) : _data(N) {}
  explicit Vector() {}

  // size
  constexpr size_t size() const noexcept { return _data.size(); }
  constexpr bool is_alloc() const noexcept { return size() > 0; }
  constexpr bool alloc(size_t N) {
    if (is_alloc()) [[unlikely]]
      return false;
    _data.resize(N);
    return true;
  }

  // data access
  constexpr T* data() noexcept { return _data.data(); }
  constexpr const T* data() const noexcept { return _data.data(); }
  constexpr T& operator[](size_t i) noexcept {
    assert(i < size());
    return _data[i];
  }
  constexpr const T& operator[](size_t i) const noexcept {
    assert(i < size());
    return _data[i];
  }

  // iterator access
  constexpr const T* begin() const noexcept { return data(); }
  constexpr const T* end() const noexcept { return data() + size(); }
  constexpr T* begin() noexcept { return data(); }
  constexpr T* end() noexcept { return data() + size(); }

  constexpr Vector& operator=(const Vector& src) = default;

  template <typename ExprType>
  constexpr Vector& operator=(const Expr<ExprType, T>& src) {
    for (size_t i = 0; i < size(); i++) {
      data()[i] = src[i];
    }
  }
};

//
// concept vectorlike
//

template <typename V>
concept VectorLike = requires(V v) {
  typename V::value_type;

  { v.size() } -> std::convertible_to<size_t>;
  { v[0] } -> std::convertible_to<typename V::value_type>;
  { v.data() };
  { v.begin() };
  { v.end() };
};

//
// operators
//

template <VectorLike V1, VectorLike V2>
constexpr auto operator+(const V1& a, const V2& b) {
  assert(a.size() == b.size());
  return BinaryExpr<V1, V2, Add, typename V1::value_type>{a, b};
}

template <VectorLike V1, VectorLike V2>
constexpr auto operator-(const V1& a, const V2& b) {
  assert(a.size() == b.size());
  return BinaryExpr<V1, V2, Sub, typename V1::value_type>{a, b};
}

template <VectorLike V1, VectorLike V2>
constexpr auto operator*(const V1& a, const V2& b) {
  assert(a.size() == b.size());
  return BinaryExpr<V1, V2, Mul, typename V1::value_type>{a, b};
}

template <VectorLike V1, VectorLike V2>
constexpr auto operator/(const V1& a, const V2& b) {
  assert(a.size() == b.size());
  return BinaryExpr<V1, V2, Div, typename V1::value_type>{a, b};
}

// other common operators

template <size_t N, typename T>
T norm2(const StaticVector<N, T>& x) {
  if constexpr (std::is_same_v<T, double>) {
    cblas_dnrm2(N, x.data(), 1);
  } else if constexpr (std::is_same_v<T, float>) {
    cblas_snrm2(N, x.data(), 1);
  } else {
    T nrm{};
    for (size_t i = 0; i < x.size(); i++) {
      nrm += x[i] * x[i];
    }
    return std::sqrt(nrm);
  }
}

template <typename T>
T norm2(const Vector<T>& x) {
  if constexpr (std::is_same_v<T, double>) {
    cblas_dnrm2(x.size(), x.data(), 1);
  } else if constexpr (std::is_same_v<T, float>) {
    cblas_snrm2(x.size(), x.data(), 1);
  } else {
    T nrm{};
    for (size_t i = 0; i < x.size(); i++) {
      nrm += x[i] * x[i];
    }
    return std::sqrt(nrm);
  }
}

// some wrapper types
#define SIZE_CONDITIONED_VECTOR_T \
  std::conditional_t<(N > 0), StaticVector<N, T>, Vector<T>>  // no-semi!

// vector with possibly custom vector norm
template <size_t N = 0, std::floating_point T = double>  //
struct NormedVector {
  SIZE_CONDITIONED_VECTOR_T v;
  T (*norm)(const SIZE_CONDITIONED_VECTOR_T& x) = norm2;
};
