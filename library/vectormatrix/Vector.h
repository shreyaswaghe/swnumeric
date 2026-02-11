#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <vector>

#include "library/expression/Expression.h"

template <size_t N, std::floating_point T>
struct StaticVector : Expr<StaticVector<N, T>, T> {
  static_assert(N > 0, "Length to StaticVector must be positive.");

  //
  using value_type = T;
  T _data[N];

  // size
  constexpr size_t size() const noexcept { return N; }

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

template <std::floating_point T>
struct Vector : Expr<Vector<T>, T> {
  //
  using value_type = T;
  std::vector<T> _data;

  // constructor
  explicit Vector(size_t N) : _data(N) {}

  // size
  constexpr size_t size() const noexcept { return _data.size(); }

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
