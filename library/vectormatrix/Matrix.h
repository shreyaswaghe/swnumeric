#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include "library/expression/Expression.h"

// Row major
template <size_t R, size_t C, typename T>
struct StaticMatrix : Expr<StaticMatrix<R, C, T>, T> {
  static_assert(R > 0, "Rows to StaticMatrix must be positive.");
  static_assert(C > 0, "Cols to StaticMatrix must be positive.");
  static_assert(std::is_floating_point_v<T>,
                "StaticMatrix is only valid for floating point types.");

  //
  using value_type = T;
  static constexpr size_t N = R * C;
  T _data[N];

  // size
  constexpr size_t size() const noexcept { return N; }
  constexpr size_t rows() const noexcept { return R; }
  constexpr size_t cols() const noexcept { return C; }

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
  constexpr StaticMatrix& operator=(const StaticMatrix& src) = default;

  template <typename ExprType>
  constexpr StaticMatrix& operator=(const Expr<ExprType, T>& src) {
    for (size_t i = 0; i < size(); i++) {
      data()[i] = src[i];
    }
  }
};

// Row major
template <typename T>
struct Matrix : Expr<Matrix<T>, T> {
  static_assert(std::is_floating_point_v<T>,
                "Matrix is only valid for floating point types.");

  //
  using value_type = T;
  std::vector<T> _data;
  size_t _R, _C;

  // constructor
  explicit Matrix(size_t R, size_t C) : _data(R * C), _R(R), _C(C) {}

  // size
  constexpr size_t size() const noexcept { return _data.size(); }
  constexpr size_t rows() const noexcept { return _R; }
  constexpr size_t cols() const noexcept { return _C; }

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

  constexpr Matrix& operator=(const Matrix& src) = default;

  template <typename ExprType>
  constexpr Matrix& operator=(const Expr<ExprType, T>& src) {
    for (size_t i = 0; i < size(); i++) {
      data()[i] = src[i];
    }
  }
};

//
// concept vectorlike
//

template <typename M>
concept MatrixLike = requires(M m) {
  typename M::value_type;

  { m.size() } -> std::convertible_to<size_t>;
  { m.rows() } -> std::convertible_to<size_t>;
  { m.cols() } -> std::convertible_to<size_t>;
  { m[0] } -> std::convertible_to<typename M::value_type>;
  { m.data() };
  { m.begin() };
  { m.end() };
};

//
// operators
//

template <MatrixLike M1, MatrixLike M2>
constexpr auto operator+(const M1& a, const M2& b) {
  assert(a.size() == b.size());
  return BinaryExpr<M1, M2, Add, typename M1::value_type>{a, b};
}

template <MatrixLike M1, MatrixLike M2>
constexpr auto operator-(const M1& a, const M2& b) {
  assert(a.size() == b.size());
  return BinaryExpr<M1, M2, Sub, typename M1::value_type>{a, b};
}

template <MatrixLike M1, MatrixLike M2>
constexpr auto operator*(const M1& a, const M2& b) {
  assert(a.size() == b.size());
  return BinaryExpr<M1, M2, Mul, typename M1::value_type>{a, b};
}

template <MatrixLike M1, MatrixLike M2>
constexpr auto operator/(const M1& a, const M2& b) {
  assert(a.size() == b.size());
  return BinaryExpr<M1, M2, Div, typename M1::value_type>{a, b};
}
