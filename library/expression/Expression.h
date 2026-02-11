#include <cstddef>

template <typename Derived, typename T>
struct Expr {
  constexpr T operator[](size_t i) const noexcept {
    return static_cast<const Derived&>(*this)[i];
  }
};

template <typename LHS, typename RHS, typename Op, typename T>
struct BinaryExpr : Expr<BinaryExpr<LHS, RHS, Op, T>, T> {
  const LHS& lhs;
  const RHS& rhs;

  constexpr T operator[](size_t i) const { return Op::apply(lhs[i], rhs[i]); }
};

// Atomic operations

struct Add {
  template <typename T>
  constexpr static T apply(T a, T b) noexcept {
    return a + b;
  }
};

struct Sub {
  template <typename T>
  constexpr static T apply(T a, T b) noexcept {
    return a - b;
  }
};

struct Mul {
  template <typename T>
  constexpr static T apply(T a, T b) noexcept {
    return a * b;
  }
};

struct Div {
  template <typename T>
  constexpr static T apply(T a, T b) noexcept {
    return a / b;
  }
};
