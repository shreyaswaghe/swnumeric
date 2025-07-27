#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

// your blas lib

#include "armpl.h"

namespace swnumeric {

namespace impl {
constexpr size_t alignment = 16;

template <typename T, size_t size>
struct ContiguousStorage {
	static_assert(size > 0, "Size must be greater than 0");

	static constexpr size_t _len = size;

	T _data[size] = {};

	bool alloc(const size_t len) { return true; }

	bool isAlloc() { return true; }

	ContiguousStorage() = default;
	ContiguousStorage(const size_t len) {};

	T* ptr() { return _data; }
	const T* ptr() const { return _data; }
	size_t len() const { return size; }
	size_t alloced_size() const { return size; }
};

// Specialization for size = 0 (dynamic allocation)
template <typename T>
struct ContiguousStorage<T, 0> {
	size_t _len;
	size_t _alloced_size;
	std::unique_ptr<T, void (*)(void*)> _data;

	bool alloc(const size_t len) {
		_len = len;
		_alloced_size =
			alignment * ((len * sizeof(T) + alignment - 1) / alignment);

		void* alloced_memory = (std::aligned_alloc(alignment, _alloced_size));
		if (!alloced_memory) {
			return false;
		}

		_data.reset(static_cast<T*>(alloced_memory));
		return true;
	}

	bool isAlloc() { return static_cast<bool>(_len); }

	ContiguousStorage()
		: _len(0), _alloced_size(0), _data(nullptr, std::free) {}

	ContiguousStorage(const size_t len)
		: _len(0), _alloced_size(0), _data(nullptr, std::free) {
		alloc(len);
		std::memset(_data.get(), 0, sizeof(T) * len);
	}

	T* ptr() { return _data.get(); }
	const T* ptr() const { return _data.get(); }
	size_t len() const { return _len; }
	size_t alloced_size() const { return _alloced_size; }
};
}  // namespace impl

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
///
///               CLASS DEFINITIONS
///
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// class declarations
template <size_t _size = 0, typename T = double>
struct Vector;

template <size_t M = 0, size_t N = 0, typename T = double>
struct Matrix;

using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector4 = Vector<4>;
using Vector5 = Vector<5>;
using Vector6 = Vector<6>;

using Matrix22 = Matrix<2, 2>;
using Matrix23 = Matrix<2, 3>;
using Matrix33 = Matrix<3, 3>;
using Matrix32 = Matrix<3, 2>;
using Matrix31 = Matrix<3, 1>;
using Matrix44 = Matrix<4, 4>;
using Matrix55 = Matrix<5, 5>;
using Matrix66 = Matrix<6, 6>;

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
///
///              IMPLEMENTATION DETAILS
///
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// operation helper definitions
namespace impl {

enum class OPType { Add, Sub, SubLeft, Mul, Div, DivLeft, Assign };

template <typename T, size_t sa, OPType _op>
struct SVOP {
	const Vector<sa, T>& vec;
	const T scalar;
	static constexpr OPType op = _op;
};

template <typename T, size_t sa, size_t sb, OPType _op>
struct VVOP {
	const Vector<sa, T>& lhs;
	const Vector<sb, T>& rhs;
	static constexpr OPType op = _op;
};

template <typename T, size_t ra, size_t ca, OPType _op>
struct SMOP {
	const Matrix<ra, ca, T>& mat;
	const T scalar;
	static constexpr OPType op = _op;
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb, OPType _op>
struct MMOP {
	const Matrix<ra, ca, T>& lhs;
	const Matrix<rb, cb, T>& rhs;
	static constexpr OPType op = _op;
};

template <typename T, size_t sa, OPType _op, OPType _destop>
struct SVOPImpl;

template <typename T, size_t ra, size_t ca, OPType _op, OPType _destop>
struct SMOPImpl;

template <typename T, size_t sa, size_t sb, OPType _op, OPType _destop>
struct VVOPImpl;

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb, OPType _op,
		  OPType _destop>
struct MMOPImpl;

}  // namespace impl

// class definitions
template <size_t _size, typename T>
struct Vector : public impl::ContiguousStorage<T, _size> {
	using impl::ContiguousStorage<T, _size>::ptr;
	using impl::ContiguousStorage<T, _size>::len;

   public:
	// pointer access
	T* operator()() { return ptr(); }
	T* operator()(size_t i) { return ptr() + i; }
	const T* operator()() const { return ptr(); }
	const T* operator()(size_t i) const { return ptr() + i; }

	// element access
	T& operator[](size_t i) { return ptr()[i]; }
	const T& operator[](size_t i) const { return ptr()[i]; }

	// size
	size_t size() const { return len(); }

	// simple setters
	void setOne() { std::fill_n(ptr(), len(), T(1.0)); }
	void setZero() { std::fill_n(ptr(), len(), T(0.0)); }
	void setConstant(const T val) { std::fill_n(ptr(), len(), T(val)); }

	// allocation
	void alloc(const size_t len) {
		impl::ContiguousStorage<T, _size>::alloc(len);
	}

	// fancy constructors
	Vector() : impl::ContiguousStorage<T, _size>() {}

	Vector(const size_t len) : impl::ContiguousStorage<T, _size>(len) {}

	Vector(std::array<T, _size>& arr)
		: swnumeric::impl::ContiguousStorage<T, _size>() {};

	Vector(const Vector<_size, T>& other) : Vector(other.size()) {
		*this = other;
	};

	// operator=
	template <size_t __size>
	Vector<_size, T>& operator=(const Vector<__size, T>& other) {
		if (size() != other.size())
			throw std::runtime_error("SIZE MISMATCH IN ASSIGNMENT " +
									 std::to_string(size()) + " and " +
									 std::to_string(other.size()));

		std::memcpy(ptr(), other(), size() * sizeof(T));
		return *this;
	};

	Vector<_size, T>& operator=(const Vector<_size, T>& other) {
		if (size() != other.size())
			throw std::runtime_error("SIZE MISMATCH IN ASSIGNMENT " +
									 std::to_string(size()) + " and " +
									 std::to_string(other.size()));

		std::memcpy(ptr(), other(), size() * sizeof(T));
		return *this;
	};

	template <size_t sa, impl::OPType _op>
	Vector<_size, T>& operator=(const impl::SVOP<T, sa, _op>& exp);
	template <size_t sa, size_t sb, impl::OPType _op>
	Vector<_size, T>& operator=(const impl::VVOP<T, sa, sb, _op>& exp);

	template <size_t sa>
	Vector<_size, T>& operator+=(const Vector<sa, T>& a);
	Vector<_size, T>& operator+=(const T& a);
	template <size_t sa, impl::OPType _op>
	Vector<_size, T>& operator+=(const impl::SVOP<T, sa, _op>& a);
	template <size_t sa, size_t sb, impl::OPType _op>
	Vector<_size, T>& operator+=(const impl::VVOP<T, sa, sb, _op>& a);

	template <size_t sa>
	Vector<_size, T>& operator-=(const Vector<sa, T>& a);
	Vector<_size, T>& operator-=(const T& a);
	template <size_t sa, impl::OPType _op>
	Vector<_size, T>& operator-=(const impl::SVOP<T, sa, _op>& a);
	template <size_t sa, size_t sb, impl::OPType _op>
	Vector<_size, T>& operator-=(const impl::VVOP<T, sa, sb, _op>& a);

	template <size_t sa>
	Vector<_size, T>& operator*=(const Vector<sa, T>& a);
	Vector<_size, T>& operator*=(const T& a);
	template <size_t sa, impl::OPType _op>
	Vector<_size, T>& operator*=(const impl::SVOP<T, sa, _op>& a);
	template <size_t sa, size_t sb, impl::OPType _op>
	Vector<_size, T>& operator*=(const impl::VVOP<T, sa, sb, _op>& a);

	template <size_t sa>
	Vector<_size, T>& operator/=(const Vector<sa, T>& a);
	Vector<_size, T>& operator/=(const T& a);
	template <size_t sa, impl::OPType _op>
	Vector<_size, T>& operator/=(const impl::SVOP<T, sa, _op>& a);
	template <size_t sa, size_t sb, impl::OPType _op>
	Vector<_size, T>& operator/=(const impl::VVOP<T, sa, sb, _op>& a);

	// cast as row matrix
	Matrix<1, _size, T> asRowMatrix() {
		Matrix<1, _size, T> x(1, size());
		std::memcpy(x(), ptr(), size() * sizeof(T));
	}
};

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
///
///               OPERATOR ENABLEMENT - VECTOR
///
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#define LINALGSIZEERROR "ERROR IN SIZE CHECK"

#define _LINALG_VECVEC_SIZECHECK \
	if (a.size() != b.size()) throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_SELFVEC_SIZECHECK \
	if (this->size() != a.size()) throw std::runtime_error(LINALGSIZEERROR);

// RETURN COMPOUND VECTOR EXPRESSION

template <typename T, size_t sa>
auto operator+(const Vector<sa, T>& a, const T scalar) {
	return impl::SVOP<T, sa, impl::OPType::Add>{.vec = a, .scalar = scalar};
}

template <typename T, size_t sa>
auto operator+(const T scalar, const Vector<sa, T>& a) {
	return impl::SVOP<T, sa, impl::OPType::Add>{.vec = a, .scalar = scalar};
}

template <typename T, size_t sa>
auto operator-(const Vector<sa, T>& a, const T scalar) {
	return impl::SVOP<T, sa, impl::OPType::Sub>{.vec = a, .scalar = scalar};
}

template <typename T, size_t sa>
auto operator-(const T scalar, const Vector<sa, T>& a) {
	return impl::SVOP<T, sa, impl::OPType::SubLeft>{.vec = a, .scalar = scalar};
}

template <typename T, size_t sa>
auto operator*(const Vector<sa, T>& a, T scalar) {
	return impl::SVOP<T, sa, impl::OPType::Mul>{.vec = a, .scalar = scalar};
}

template <typename T, size_t sa>
auto operator*(const T scalar, const Vector<sa, T>& a) {
	return impl::SVOP<T, sa, impl::OPType::Mul>{.vec = a, .scalar = scalar};
}

template <typename T, size_t sa>
auto operator/(const Vector<sa, T>& a, const T scalar) {
	return impl::SVOP<T, sa, impl::OPType::Div>{.vec = a, .scalar = scalar};
}

template <typename T, size_t sa>
auto operator/(const T scalar, const Vector<sa, T>& a) {
	return impl::SVOP<T, sa, impl::OPType::DivLeft>{.vec = a, .scalar = scalar};
}

template <typename T, size_t sa, size_t sb>
auto operator+(const Vector<sa, T>& a, const Vector<sb, T>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return impl::VVOP<T, sa, sb, impl::OPType::Add>{.lhs = a, .rhs = b};
}

template <typename T, size_t sa, size_t sb>
auto operator-(const Vector<sa, T>& a, const Vector<sb, T>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return impl::VVOP<T, sa, sb, impl::OPType::Sub>{.lhs = a, .rhs = b};
}

template <typename T, size_t sa, size_t sb>
auto operator*(const Vector<sa, T>& a, const Vector<sb, T>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return impl::VVOP<T, sa, sb, impl::OPType::Mul>{.lhs = a, .rhs = b};
}

template <typename T, size_t sa, size_t sb>
auto operator/(const Vector<sa, T>& a, const Vector<sb, T>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return impl::VVOP<T, sa, sb, impl::OPType::Div>{.lhs = a, .rhs = b};
}

// DISPATCH TO COMPUTATIONAL KERNELS ON ASSIGNMENT
// ADDITION
template <size_t __size, typename T>
template <size_t sa>
Vector<__size, T>& Vector<__size, T>::operator+=(const Vector<sa, T>& a) {
	add(*this, a);
	return *this;
}

template <size_t __size, typename T>
Vector<__size, T>& Vector<__size, T>::operator+=(const T& a) {
	add(*this, a);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator+=(
	const impl::SVOP<T, sa, _op>& exp) {
	impl::SVOPImpl<T, sa, _op, impl::OPType::Add>::apply(*this, exp);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, size_t sb, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator+=(
	const impl::VVOP<T, sa, sb, _op>& exp) {
	impl::VVOPImpl<T, sa, sb, _op, impl::OPType::Add>::apply(*this, exp);
	return *this;
}

// SUBTRACTION
template <size_t __size, typename T>
template <size_t sa>
Vector<__size, T>& Vector<__size, T>::operator-=(const Vector<sa, T>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	sub(*this, a);
	return *this;
}

template <size_t __size, typename T>
Vector<__size, T>& Vector<__size, T>::operator-=(const T& a) {
	sub(*this, a);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator-=(
	const impl::SVOP<T, sa, _op>& exp) {
	impl::SVOPImpl<T, sa, _op, impl::OPType::Sub>::apply(*this, exp);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, size_t sb, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator-=(
	const impl::VVOP<T, sa, sb, _op>& exp) {
	impl::VVOPImpl<T, sa, sb, _op, impl::OPType::Sub>::apply(*this, exp);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa>
Vector<__size, T>& Vector<__size, T>::operator*=(const Vector<sa, T>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	mul(*this, a);
	return *this;
}

template <size_t __size, typename T>
Vector<__size, T>& Vector<__size, T>::operator*=(const T& a) {
	mul(*this, a);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator*=(
	const impl::SVOP<T, sa, _op>& exp) {
	impl::SVOPImpl<T, sa, _op, impl::OPType::Mul>::apply(*this, exp);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, size_t sb, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator*=(
	const impl::VVOP<T, sa, sb, _op>& exp) {
	impl::VVOPImpl<T, sa, sb, _op, impl::OPType::Mul>::apply(*this, exp);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa>
Vector<__size, T>& Vector<__size, T>::operator/=(const Vector<sa, T>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	div(*this, a);
	return *this;
}

template <size_t __size, typename T>
Vector<__size, T>& Vector<__size, T>::operator/=(const T& a) {
	div(*this, a);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator/=(
	const impl::SVOP<T, sa, _op>& exp) {
	impl::SVOPImpl<T, sa, _op, impl::OPType::Div>::apply(*this, exp);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, size_t sb, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator/=(
	const impl::VVOP<T, sa, sb, _op>& exp) {
	impl::VVOPImpl<T, sa, sb, _op, impl::OPType::Div>::apply(*this, exp);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator=(
	const impl::SVOP<T, sa, _op>& exp) {
	impl::SVOPImpl<T, sa, _op, impl::OPType::Assign>::apply(*this, exp);
	return *this;
}

template <size_t __size, typename T>
template <size_t sa, size_t sb, impl::OPType _op>
Vector<__size, T>& Vector<__size, T>::operator=(
	const impl::VVOP<T, sa, sb, _op>& exp) {
	impl::VVOPImpl<T, sa, sb, _op, impl::OPType::Assign>::apply(*this, exp);
	return *this;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
///
/// NOBLAS ATOMIC OPERATIONS FOR VECTOR
///
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template <typename T, size_t sa, size_t sb>
void add(Vector<sa, T>& a, const Vector<sb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
}

template <typename T, size_t sa>
void add(Vector<sa, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] += b;
}

template <typename T, size_t sa, size_t sb>
void sub(Vector<sa, T>& a, const Vector<sb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] -= b[i];
}

template <typename T, size_t sa>
void sub(Vector<sa, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] -= b;
}

template <typename T, size_t sa, size_t sb>
void subLeft(Vector<sa, T>& a, const Vector<sb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b[i] - a[i];
}

template <typename T, size_t sa>
void subLeft(Vector<sa, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b - a[i];
}

template <typename T, size_t sa, size_t sb>
void mul(Vector<sa, T>& a, const Vector<sb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] *= b[i];
}

template <typename T, size_t sa>
void mul(Vector<sa, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] *= b;
}

template <typename T, size_t sa, size_t sb>
void div(Vector<sa, T>& a, const Vector<sb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] /= b[i];
}

template <typename T, size_t sa>
void div(Vector<sa, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] /= b;
}

template <typename T, size_t sa, size_t sb>
void divLeft(Vector<sa, T>& a, const Vector<sb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b[i] / a[i];
}

template <typename T, size_t sa>
void divLeft(Vector<sa, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b / a[i];
}

template <typename T, size_t sa, size_t sb>
void copyTo(Vector<sa, T>& a, const Vector<sb, T>& b) {
	std::memcpy(a(), b(), a.size() * sizeof(T));
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
///
/// COMPUTATIONAL KERNEL DEFINITIONS
///
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

namespace impl {

// WHERE RESULT IS ADDED TO DEST

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
		add(dest, exp.vec);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
		add(dest, exp.vec);
		sub(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
		sub(dest, exp.vec);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
		sub(dest, exp.vec);
		sub(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
		sub(dest, exp.vec);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
		add(dest, exp.vec);
		sub(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], dest[i] * exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], -dest[i] * exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], -exp.vec[i], dest[i] * exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
		div(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] + exp.scalar;
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] - exp.scalar;
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.scalar - exp.vec[i];
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
		mul(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
		copyTo(dest, exp.vec);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
		copyTo(dest, exp.vec);
		sub(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
		copyTo(dest, exp.vec);
		mul(dest, -1.0);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
		copyTo(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
		copyTo(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

template <typename T, size_t sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
		dest.setOne();
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

// WHERE RESULT IS ADDED TO DEST

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
		add(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
		add(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(1.0 / exp.rhs[i], exp.lhs[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
		sub(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
		sub(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-1.0 / exp.rhs[i], exp.lhs[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], dest[i] * exp.rhs[i]);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], -dest[i] * exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
		mul(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
		div(dest, exp.rhs);
		mul(dest, exp.lhs);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] + exp.rhs[i];
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] - exp.rhs[i];
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
		div(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
		mul(dest, exp.rhs);
		div(dest, exp.lhs);
	};
};

// WHERE RESULT IS ASSIGNED TO DEST
template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
		copyTo(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
		copyTo(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
		copyTo(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <typename T, size_t sa, size_t sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, T>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
		copyTo(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

//
//  DOUBLE SPECIALIZATION
//

// WHERE RESULT IS ADDED TO DEST
template <size_t sa>
struct SVOPImpl<double, sa, OPType::Add, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Add>& exp) {
		cblas_daxpy(exp.vec.size(), 1.0, exp.vec(), 1, dest(), 1);

		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Sub, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Sub>& exp) {
		cblas_daxpy(dest.size(), 1.0, exp.vec(), 1, dest(), 1);

		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::SubLeft, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::SubLeft>& exp) {
		cblas_daxpy(dest.size(), -1.0, exp.vec(), 1, dest(), 1);

		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Mul, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Div, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::DivLeft, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t sa>
struct SVOPImpl<double, sa, OPType::Add, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Add>& exp) {
		cblas_daxpy(exp.vec.size(), -1.0, exp.vec(), 1, dest(), 1);
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Sub, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Sub>& exp) {
		cblas_daxpy(exp.vec.size(), -1.0, exp.vec(), 1, dest(), 1);
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::SubLeft, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::SubLeft>& exp) {
		cblas_daxpy(exp.vec.size(), 1.0, exp.vec(), 1, dest(), 1);
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Mul, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Div, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::DivLeft, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t sa>
struct SVOPImpl<double, sa, OPType::Add, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], dest[i] * exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Sub, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], -dest[i] * exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::SubLeft, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], -exp.vec[i], dest[i] * exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Mul, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Mul>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Div, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Div>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::DivLeft, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::DivLeft>& exp) {
		div(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t sa>
struct SVOPImpl<double, sa, OPType::Add, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] + exp.scalar;
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Sub, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] - exp.scalar;
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::SubLeft, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.scalar - exp.vec[i];
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Mul, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Mul>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Div, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Div>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::DivLeft, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::DivLeft>& exp) {
		mul(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <size_t sa>
struct SVOPImpl<double, sa, OPType::Add, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Add>& exp) {
		dest = exp.vec;
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Sub, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Sub>& exp) {
		dest = exp.vec;
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::SubLeft, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::SubLeft>& exp) {
		dest = exp.vec;
		mul(dest, -1.0);
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Mul, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Mul>& exp) {
		dest = exp.vec;
		mul(dest, exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::Div, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::Div>& exp) {
		dest = exp.vec;
		div(dest, exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<double, sa, OPType::DivLeft, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const SVOP<double, sa, OPType::DivLeft>& exp) {
		dest.setConstant(exp.scalar);
		div(dest, exp.vec);
	};
};

// WHERE RESULT IS ADDED TO DEST

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Add, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Add>& exp) {
		cblas_daxpy(exp.lhs.size(), 1.0, exp.lhs(), 1, dest(), 1);
		cblas_daxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Sub, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Sub>& exp) {
		cblas_daxpy(exp.lhs.size(), 1.0, exp.lhs(), 1, dest(), 1);
		cblas_daxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Mul, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Div, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], 1.0 / exp.rhs[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Add, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Add>& exp) {
		cblas_daxpy(exp.lhs.size(), -1.0, exp.lhs(), 1, dest(), 1);
		cblas_daxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Sub, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Sub>& exp) {
		cblas_daxpy(exp.lhs.size(), -1.0, exp.lhs(), 1, dest(), 1);
		cblas_daxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Mul, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Div, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], 1.0 / exp.rhs[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Add, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], dest[i] * exp.rhs[i]);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Sub, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], -dest[i] * exp.rhs[i]);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Mul, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Mul>& exp) {
		mul(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Div, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Div>& exp) {
		div(dest, exp.rhs);
		mul(dest, exp.lhs);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Add, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] + exp.rhs[i];
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Sub, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] - exp.rhs[i];
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Mul, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Mul>& exp) {
		div(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Div, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Div>& exp) {
		mul(dest, exp.rhs);
		div(dest, exp.lhs);
	};
};

// WHERE RESULT IS ASSIGNED TO DEST
template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Add, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Add>& exp) {
		dest = exp.lhs;
		cblas_daxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Sub, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Sub>& exp) {
		dest = exp.lhs;
		cblas_daxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Mul, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Mul>& exp) {
		dest = exp.lhs;
		mul(dest, exp.rhs);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<double, sa, sb, OPType::Div, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, double>& dest,
					  const VVOP<double, sa, sb, OPType::Div>& exp) {
		dest = exp.lhs;
		div(dest, exp.rhs);
	};
};

///
///   FLOAT SPECIALIZATION
///

// WHERE RESULT IS ADDED TO DEST
template <size_t sa>
struct SVOPImpl<float, sa, OPType::Add, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Add>& exp) {
		cblas_saxpy(exp.vec.size(), 1.0, exp.vec(), 1, dest(), 1);

		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Sub, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Sub>& exp) {
		cblas_saxpy(dest.size(), 1.0, exp.vec(), 1, dest(), 1);

		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::SubLeft, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::SubLeft>& exp) {
		cblas_saxpy(dest.size(), -1.0, exp.vec(), 1, dest(), 1);

		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Mul, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Div, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::DivLeft, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t sa>
struct SVOPImpl<float, sa, OPType::Add, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Add>& exp) {
		cblas_saxpy(exp.vec.size(), -1.0, exp.vec(), 1, dest(), 1);
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Sub, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Sub>& exp) {
		cblas_saxpy(exp.vec.size(), -1.0, exp.vec(), 1, dest(), 1);
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::SubLeft, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::SubLeft>& exp) {
		cblas_saxpy(exp.vec.size(), 1.0, exp.vec(), 1, dest(), 1);
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Mul, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Div, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::DivLeft, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t sa>
struct SVOPImpl<float, sa, OPType::Add, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], dest[i] * exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Sub, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], -dest[i] * exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::SubLeft, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], -exp.vec[i], dest[i] * exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Mul, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Mul>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Div, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Div>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::DivLeft, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::DivLeft>& exp) {
		div(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t sa>
struct SVOPImpl<float, sa, OPType::Add, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] + exp.scalar;
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Sub, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] - exp.scalar;
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::SubLeft, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.scalar - exp.vec[i];
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Mul, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Mul>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Div, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Div>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::DivLeft, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::DivLeft>& exp) {
		mul(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <size_t sa>
struct SVOPImpl<float, sa, OPType::Add, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Add>& exp) {
		dest = exp.vec;
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Sub, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Sub>& exp) {
		dest = exp.vec;
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::SubLeft, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::SubLeft>& exp) {
		dest = exp.vec;
		mul(dest, -1.0);
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Mul, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Mul>& exp) {
		dest = exp.vec;
		mul(dest, exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::Div, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::Div>& exp) {
		dest = exp.vec;
		div(dest, exp.scalar);
	};
};

template <size_t sa>
struct SVOPImpl<float, sa, OPType::DivLeft, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const SVOP<float, sa, OPType::DivLeft>& exp) {
		dest.setConstant(exp.scalar);
		div(dest, exp.vec);
	};
};

// WHERE RESULT IS ADDED TO DEST

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Add, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Add>& exp) {
		cblas_saxpy(exp.lhs.size(), 1.0, exp.lhs(), 1, dest(), 1);
		cblas_saxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Sub, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Sub>& exp) {
		cblas_saxpy(exp.lhs.size(), 1.0, exp.lhs(), 1, dest(), 1);
		cblas_saxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Mul, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Div, OPType::Add> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], 1.0 / exp.rhs[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Add, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Add>& exp) {
		cblas_saxpy(exp.lhs.size(), -1.0, exp.lhs(), 1, dest(), 1);
		cblas_saxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Sub, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Sub>& exp) {
		cblas_saxpy(exp.lhs.size(), -1.0, exp.lhs(), 1, dest(), 1);
		cblas_saxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Mul, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Div, OPType::Sub> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], 1.0 / exp.rhs[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Add, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], dest[i] * exp.rhs[i]);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Sub, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], -dest[i] * exp.rhs[i]);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Mul, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Mul>& exp) {
		mul(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Div, OPType::Mul> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Div>& exp) {
		div(dest, exp.rhs);
		mul(dest, exp.lhs);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Add, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] + exp.rhs[i];
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Sub, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] - exp.rhs[i];
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Mul, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Mul>& exp) {
		div(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Div, OPType::Div> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Div>& exp) {
		mul(dest, exp.rhs);
		div(dest, exp.lhs);
	};
};

// WHERE RESULT IS ASSIGNED TO DEST
template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Add, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Add>& exp) {
		dest = exp.lhs;
		cblas_saxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Sub, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Sub>& exp) {
		dest = exp.lhs;
		cblas_saxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Mul, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Mul>& exp) {
		dest = exp.lhs;
		mul(dest, exp.rhs);
	};
};

template <size_t sa, size_t sb>
struct VVOPImpl<float, sa, sb, OPType::Div, OPType::Assign> {
	template <size_t __size>
	static void apply(Vector<__size, float>& dest,
					  const VVOP<float, sa, sb, OPType::Div>& exp) {
		dest = exp.lhs;
		div(dest, exp.rhs);
	};
};

}  // namespace impl

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
///
///     MATRIX
///
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

template <size_t _rows, size_t _cols, typename T>
struct Matrix : public impl::ContiguousStorage<T, _rows * _cols> {
	static constexpr size_t _size = _rows * _cols;
	size_t Rows = _rows;
	size_t Cols = _cols;

	using impl::ContiguousStorage<T, _size>::ptr;
	using impl::ContiguousStorage<T, _size>::len;

   public:
	// pointer access
	T* operator()() { return ptr(); }
	T* operator()(size_t i) { return ptr() + i; }
	T* operator()(size_t i, size_t j) { return ptr() + i + j * lda(); }
	const T* operator()() const { return ptr(); }
	const T* operator()(size_t i) const { return ptr() + i; }
	const T* operator()(size_t i, size_t j) const {
		return ptr() + i + j * lda();
	}

	// element access
	T& operator[](size_t i) { return ptr()[i]; }
	const T& operator[](size_t i) const { return ptr()[i]; }

	// size
	size_t size() const { return len(); }
	size_t lda() const { return rows(); }
	size_t rows() const { return Rows; }
	size_t cols() const { return Cols; }

	// simple setters
	void setOne() { std::fill_n(ptr(), len(), T(1.0)); }
	void setZero() { std::fill_n(ptr(), len(), T(0.0)); }
	void setConstant(const T val) { std::fill_n(ptr(), len(), T(val)); }
	void setIdentity() {
		size_t n = std::min(rows(), cols());
		for (size_t i = 0; i < n; i++) {
			ptr()[i + i * lda()] = T(1.0);
		}
	}

	// allocation
	bool alloc(const size_t rows, const size_t cols) {
		return impl::ContiguousStorage<T, _size>::alloc(rows * cols);
	}

	// fancy constructors
	Matrix() : impl::ContiguousStorage<T, _size>() {}

	Matrix(const size_t rows, const size_t cols)
		: impl::ContiguousStorage<T, _size>(rows * cols) {
		if constexpr (_size == 0) {
			Rows = rows;
			Cols = cols;
		}
	}

	Matrix(const std::array<std::array<T, _cols>, _rows>& arr)
		: swnumeric::impl::ContiguousStorage<T, _size>() {
		for (size_t i = 0; i < _rows; i++) {
			for (size_t j = 0; j < _cols; j++) {
				ptr()[i + j * lda()] = arr[i][j];
			}
		}
	};

	Matrix(const Matrix<_rows, _cols, T>& other)
		: Matrix(other.rows(), other.cols()) {
		*this = other;
	};

	// operator=
	template <size_t __rows, size_t __cols>
	Matrix<_rows, _cols, T>& operator=(const Matrix<__rows, __cols, T>& other) {
		if (rows() != other.rows() || cols() != other.cols())
			throw std::runtime_error("SIZE MISMATCH IN ASSIGNMENT (" +
									 std::to_string(rows()) + "," +
									 std::to_string(cols()) + ") and (" +
									 std::to_string(other.rows()) + ", " +
									 std::to_string(other.cols()) + ")");

		std::memcpy(ptr(), other(), size() * sizeof(T));
		return *this;
	};

	Matrix<_rows, _cols, T>& operator=(const Matrix<_rows, _cols, T>& other) {
		if (rows() != other.rows() || cols() != other.cols())
			throw std::runtime_error("SIZE MISMATCH IN ASSIGNMENT (" +
									 std::to_string(rows()) + "," +
									 std::to_string(cols()) + ") and (" +
									 std::to_string(other.rows()) + ", " +
									 std::to_string(other.cols()) + ")");

		std::memcpy(ptr(), other(), size() * sizeof(T));
		return *this;
	};

	template <size_t ra, size_t ca, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator=(const impl::SMOP<T, ra, ca, _op>& exp);
	template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator=(
		const impl::MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <size_t ra, size_t ca>
	Matrix<_rows, _cols, T>& operator+=(const Matrix<ra, ca, T>& a);
	Matrix<_rows, _cols, T>& operator+=(const T& a);
	template <size_t ra, size_t ca, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator+=(const impl::SMOP<T, ra, ca, _op>& exp);
	template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator+=(
		const impl::MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <size_t ra, size_t ca>
	Matrix<_rows, _cols, T>& operator-=(const Matrix<ra, ca, T>& a);
	Matrix<_rows, _cols, T>& operator-=(const T& a);
	template <size_t ra, size_t ca, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator-=(const impl::SMOP<T, ra, ca, _op>& exp);
	template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator-=(
		const impl::MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <size_t ra, size_t ca>
	Matrix<_rows, _cols, T>& operator*=(const Matrix<ra, ca, T>& a);
	Matrix<_rows, _cols, T>& operator*=(const T& a);
	template <size_t ra, size_t ca, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator*=(const impl::SMOP<T, ra, ca, _op>& exp);
	template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator*=(
		const impl::MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <size_t ra, size_t ca>
	Matrix<_rows, _cols, T>& operator/=(const Matrix<ra, ca, T>& a);
	Matrix<_rows, _cols, T>& operator/=(const T& a);
	template <size_t ra, size_t ca, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator/=(const impl::SMOP<T, ra, ca, _op>& exp);
	template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
	Matrix<_rows, _cols, T>& operator/=(
		const impl::MMOP<T, ra, ca, rb, cb, _op>& exp);

	// cast as vector
	Vector<_size, T> asVector() const {
		Vector<_size, T> x(size());
		std::memcpy(x(), ptr(), size() * sizeof(T));
		return x;
	}

	Vector<std::min(_rows, _cols), T> diagonal() const {
		constexpr size_t s = std::min(_rows, _cols);
		size_t sz = std::min(rows(), cols());

		Vector<s, T> x(sz);
		for (size_t i = 0; i < sz; i++) {
			x[i] = (*this)[i + lda() * i];
		}
	}
};

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
///
///  OPERATOR ENABLEMENT
///
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

template <typename T, size_t ra, size_t ca>
auto operator+(const Matrix<ra, ca, T>& a, const T scalar) {
	return impl::SMOP<T, ra, ca, impl::OPType::Add>{.mat = a, .scalar = scalar};
}

template <typename T, size_t ra, size_t ca>
auto operator+(const T scalar, const Matrix<ra, ca, T>& a) {
	return impl::SMOP<T, ra, ca, impl::OPType::Add>{.mat = a, .scalar = scalar};
}

template <typename T, size_t ra, size_t ca>
auto operator-(const Matrix<ra, ca, T>& a, const T scalar) {
	return impl::SMOP<T, ra, ca, impl::OPType::Sub>{.mat = a, .scalar = scalar};
}

template <typename T, size_t ra, size_t ca>
auto operator-(const T scalar, const Matrix<ra, ca, T>& a) {
	return impl::SMOP<T, ra, ca, impl::OPType::SubLeft>{.mat = a,
														.scalar = scalar};
}

template <typename T, size_t ra, size_t ca>
auto operator*(const Matrix<ra, ca, T>& a, T scalar) {
	return impl::SMOP<T, ra, ca, impl::OPType::Mul>{.mat = a, .scalar = scalar};
}

template <typename T, size_t ra, size_t ca>
auto operator*(const T scalar, const Matrix<ra, ca, T>& a) {
	return impl::SMOP<T, ra, ca, impl::OPType::Mul>{.mat = a, .scalar = scalar};
}

template <typename T, size_t ra, size_t ca>
auto operator/(const Matrix<ra, ca, T>& a, const T scalar) {
	return impl::SMOP<T, ra, ca, impl::OPType::Div>{.mat = a, .scalar = scalar};
}

template <typename T, size_t ra, size_t ca>
auto operator/(const T scalar, const Matrix<ra, ca, T>& a) {
	return impl::SMOP<T, ra, ca, impl::OPType::DivLeft>{.mat = a,
														.scalar = scalar};
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator+=(
	const Matrix<ra, ca, T>& a) {
	add(*this, a);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator+=(const T& a) {
	add(*this, a);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator+=(
	const impl::SMOP<T, ra, ca, _op>& exp) {
	impl::SMOPImpl<T, ra, ca, _op, impl::OPType::Add>::apply(*this, exp);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator+=(
	const impl::MMOP<T, ra, ca, rb, cb, _op>& exp) {
	impl::MMOPImpl<T, ra, ca, rb, cb, _op, impl::OPType::Add>::apply(*this,
																	 exp);
	return *this;
}

// SUBTRACTION
template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator-=(
	const Matrix<ra, ca, T>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	sub(*this, a);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator-=(const T& a) {
	sub(*this, a);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator-=(
	const impl::SMOP<T, ra, ca, _op>& exp) {
	impl::SMOPImpl<T, ra, ca, _op, impl::OPType::Sub>::apply(*this, exp);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator-=(
	const impl::MMOP<T, ra, ca, rb, cb, _op>& exp) {
	impl::MMOPImpl<T, ra, ca, rb, cb, _op, impl::OPType::Sub>::apply(*this,
																	 exp);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator*=(
	const Matrix<ra, ca, T>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	mul(*this, a);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator*=(const T& a) {
	mul(*this, a);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator*=(
	const impl::SMOP<T, ra, ca, _op>& exp) {
	impl::SMOPImpl<T, ra, ca, _op, impl::OPType::Mul>::apply(*this, exp);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator*=(
	const impl::MMOP<T, ra, ca, rb, cb, _op>& exp) {
	impl::MMOPImpl<T, ra, ca, rb, cb, _op, impl::OPType::Mul>::apply(*this,
																	 exp);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator/=(
	const Matrix<ra, ca, T>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	div(*this, a);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator/=(const T& a) {
	div(*this, a);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator/=(
	const impl::SMOP<T, ra, ca, _op>& exp) {
	impl::SMOPImpl<T, ra, ca, _op, impl::OPType::Div>::apply(*this, exp);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator/=(
	const impl::MMOP<T, ra, ca, rb, cb, _op>& exp) {
	impl::MMOPImpl<T, ra, ca, rb, cb, _op, impl::OPType::Div>::apply(*this,
																	 exp);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator=(
	const impl::SMOP<T, ra, ca, _op>& exp) {
	impl::SMOPImpl<T, ra, ca, _op, impl::OPType::Assign>::apply(*this, exp);
	return *this;
}

template <size_t __rows, size_t __cols, typename T>
template <size_t ra, size_t ca, size_t rb, size_t cb, impl::OPType _op>
Matrix<__rows, __cols, T>& Matrix<__rows, __cols, T>::operator=(
	const impl::MMOP<T, ra, ca, rb, cb, _op>& exp) {
	impl::MMOPImpl<T, ra, ca, rb, cb, _op, impl::OPType::Assign>::apply(*this,
																		exp);
	return *this;
}

#define _LINALG_MATMAT_SIZECHECK                      \
	if (a.rows() != b.rows() || a.cols() != b.cols()) \
		throw std::runtime_error(LINALGSIZEERROR);

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
auto operator+(const Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
	_LINALG_MATMAT_SIZECHECK;
	return impl::MMOP<T, ra, ca, rb, cb, impl::OPType::Add>{.lhs = a, .rhs = b};
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
auto operator-(const Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
	_LINALG_MATMAT_SIZECHECK;
	return impl::MMOP<T, ra, ca, rb, cb, impl::OPType::Sub>{.lhs = a, .rhs = b};
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
auto operator*(const Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
	_LINALG_MATMAT_SIZECHECK;
	return impl::MMOP<T, ra, ca, rb, cb, impl::OPType::Mul>{.lhs = a, .rhs = b};
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
auto operator/(const Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
	_LINALG_MATMAT_SIZECHECK;
	return impl::MMOP<T, ra, ca, rb, cb, impl::OPType::Div>{.lhs = a, .rhs = b};
}

// NO-BLAS ATOMIC OPERATIONS FOR MATRIX
template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
void add(Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
}

template <typename T, size_t ra, size_t ca>
void add(Matrix<ra, ca, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] += b;
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
void sub(Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] -= b[i];
}

template <typename T, size_t ra, size_t ca>
void sub(Matrix<ra, ca, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] -= b;
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
void subLeft(Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b[i] - a[i];
}

template <typename T, size_t ra, size_t ca>
void subLeft(Matrix<ra, ca, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b - a[i];
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
void mul(Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] *= b[i];
}

template <typename T, size_t ra, size_t ca>
void mul(Matrix<ra, ca, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] *= b;
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
void div(Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] /= b[i];
}

template <typename T, size_t ra, size_t ca>
void div(Matrix<ra, ca, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] /= b;
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
void divLeft(Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b[i] / a[i];
}

template <typename T, size_t ra, size_t ca>
void divLeft(Matrix<ra, ca, T>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b / a[i];
}

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
void copyTo(Matrix<ra, ca, T>& a, const Matrix<rb, cb, T>& b) {
	std::memcpy(a(), b(), a.size() * sizeof(T));
}

namespace impl {

#define _LINALG_MATMAT_SIZECHECK                      \
	if (a.rows() != b.rows() || a.cols() != b.cols()) \
		throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_SELFMAT_SIZECHECK                             \
	if (this->rows() != a.rows() || this->cols() != a.cols()) \
		throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_MATVEC_COMPATIBILITY_CHECK             \
	if (A.cols() != a.size() || out.size() < A.rows()) \
		throw std::runtime_error(LINALGSIZEERROR);

template <typename T, size_t ra, size_t ca, OPType _op, OPType _destop>
struct SMOPImpl;

// WHERE RESULT IS ADDED TO DEST

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
		add(dest, exp.mat);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
		add(dest, exp.mat);
		sub(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
		sub(dest, exp.mat);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.mat[i], dest[i]);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.mat[i], dest[i]);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.mat[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
		sub(dest, exp.mat);
		sub(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
		sub(dest, exp.mat);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
		add(dest, exp.mat);
		sub(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.mat[i], dest[i]);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.mat[i], dest[i]);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.mat[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], exp.mat[i], dest[i] * exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], exp.mat[i], -dest[i] * exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], -exp.mat[i], dest[i] * exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.mat);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.mat);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
		div(dest, exp.mat);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.mat[i] + exp.scalar;
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.mat[i] - exp.scalar;
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.scalar - exp.mat[i];
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
		mul(dest, exp.mat);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
		copyTo(dest, exp.mat);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
		copyTo(dest, exp.mat);
		sub(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
		copyTo(dest, exp.mat);
		mul(dest, -1.0);
		add(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
		copyTo(dest, exp.mat);
		mul(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
		copyTo(dest, exp.mat);
		div(dest, exp.scalar);
	};
};

template <typename T, size_t ra, size_t ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
		dest.setOne();
		mul(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb, OPType _op,
		  OPType _destop>
struct MMOPImpl;

// WHERE RESULT IS ADDED TO DEST

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
		add(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
		add(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(1.0 / exp.rhs[i], exp.lhs[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
		sub(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
		sub(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-1.0 / exp.rhs[i], exp.lhs[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], dest[i] * exp.rhs[i]);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], -dest[i] * exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
		mul(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
		div(dest, exp.rhs);
		mul(dest, exp.lhs);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] + exp.rhs[i];
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] - exp.rhs[i];
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
		div(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
		mul(dest, exp.rhs);
		div(dest, exp.lhs);
	};
};

// WHERE RESULT IS ASSIGNED TO DEST
template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
		copyTo(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
		copyTo(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
		copyTo(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <typename T, size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, T>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
		copyTo(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

//
//  DOUBLE SPECIALIZATION
//

// WHERE RESULT IS ADDED TO DEST
template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Add, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Add>& exp) {
		cblas_daxpy(exp.mat.size(), 1.0, exp.mat(), 1, dest(), 1);

		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Sub, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Sub>& exp) {
		cblas_daxpy(dest.size(), 1.0, exp.mat(), 1, dest(), 1);

		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::SubLeft, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::SubLeft>& exp) {
		cblas_daxpy(dest.size(), -1.0, exp.mat(), 1, dest(), 1);

		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Mul, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.mat[i], dest[i]);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Div, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.mat[i], dest[i]);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::DivLeft, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.mat[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Add, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Add>& exp) {
		cblas_daxpy(exp.mat.size(), -1.0, exp.mat(), 1, dest(), 1);
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Sub, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Sub>& exp) {
		cblas_daxpy(exp.mat.size(), -1.0, exp.mat(), 1, dest(), 1);
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::SubLeft, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::SubLeft>& exp) {
		cblas_daxpy(exp.mat.size(), 1.0, exp.mat(), 1, dest(), 1);
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Mul, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.mat[i], dest[i]);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Div, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.mat[i], dest[i]);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::DivLeft, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.mat[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Add, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], exp.mat[i], dest[i] * exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Sub, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], exp.mat[i], -dest[i] * exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::SubLeft, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], -exp.mat[i], dest[i] * exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Mul, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Mul>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.mat);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Div, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Div>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.mat);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::DivLeft, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::DivLeft>& exp) {
		div(dest, exp.mat);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Add, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.mat[i] + exp.scalar;
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Sub, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.mat[i] - exp.scalar;
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::SubLeft, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.scalar - exp.mat[i];
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Mul, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Mul>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Div, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Div>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::DivLeft, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::DivLeft>& exp) {
		mul(dest, exp.mat);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Add, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Add>& exp) {
		dest = exp.mat;
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Sub, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Sub>& exp) {
		dest = exp.mat;
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::SubLeft, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::SubLeft>& exp) {
		dest = exp.mat;
		mul(dest, -1.0);
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Mul, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Mul>& exp) {
		dest = exp.mat;
		mul(dest, exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::Div, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::Div>& exp) {
		dest = exp.mat;
		div(dest, exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<double, ra, ca, OPType::DivLeft, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const SMOP<double, ra, ca, OPType::DivLeft>& exp) {
		dest.setConstant(exp.scalar);
		div(dest, exp.mat);
	};
};

// WHERE RESULT IS ADDED TO DEST

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Add, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Add>& exp) {
		cblas_daxpy(exp.lhs.size(), 1.0, exp.lhs(), 1, dest(), 1);
		cblas_daxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Sub, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Sub>& exp) {
		cblas_daxpy(exp.lhs.size(), 1.0, exp.lhs(), 1, dest(), 1);
		cblas_daxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Mul, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Div, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], 1.0 / exp.rhs[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Add, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Add>& exp) {
		cblas_daxpy(exp.lhs.size(), -1.0, exp.lhs(), 1, dest(), 1);
		cblas_daxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Sub, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Sub>& exp) {
		cblas_daxpy(exp.lhs.size(), -1.0, exp.lhs(), 1, dest(), 1);
		cblas_daxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Mul, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Div, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], 1.0 / exp.rhs[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Add, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], dest[i] * exp.rhs[i]);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Sub, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], -dest[i] * exp.rhs[i]);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Mul, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Mul>& exp) {
		mul(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Div, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Div>& exp) {
		div(dest, exp.rhs);
		mul(dest, exp.lhs);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Add, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] + exp.rhs[i];
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Sub, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] - exp.rhs[i];
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Mul, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Mul>& exp) {
		div(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Div, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Div>& exp) {
		mul(dest, exp.rhs);
		div(dest, exp.lhs);
	};
};

// WHERE RESULT IS ASSIGNED TO DEST
template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Add, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Add>& exp) {
		dest = exp.lhs;
		cblas_daxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Sub, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Sub>& exp) {
		dest = exp.lhs;
		cblas_daxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Mul, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Mul>& exp) {
		dest = exp.lhs;
		mul(dest, exp.rhs);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<double, ra, ca, rb, cb, OPType::Div, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, double>& dest,
					  const MMOP<double, ra, ca, rb, cb, OPType::Div>& exp) {
		dest = exp.lhs;
		div(dest, exp.rhs);
	};
};

//
//  float SPECIALIZATION
//

// WHERE RESULT IS ADDED TO DEST
template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Add, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Add>& exp) {
		cblas_saxpy(exp.mat.size(), 1.0, exp.mat(), 1, dest(), 1);

		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Sub, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Sub>& exp) {
		cblas_saxpy(dest.size(), 1.0, exp.mat(), 1, dest(), 1);

		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::SubLeft, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::SubLeft>& exp) {
		cblas_saxpy(dest.size(), -1.0, exp.mat(), 1, dest(), 1);

		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Mul, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.mat[i], dest[i]);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Div, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.mat[i], dest[i]);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::DivLeft, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.mat[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Add, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Add>& exp) {
		cblas_saxpy(exp.mat.size(), -1.0, exp.mat(), 1, dest(), 1);
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Sub, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Sub>& exp) {
		cblas_saxpy(exp.mat.size(), -1.0, exp.mat(), 1, dest(), 1);
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::SubLeft, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::SubLeft>& exp) {
		cblas_saxpy(exp.mat.size(), 1.0, exp.mat(), 1, dest(), 1);
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Mul, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.mat[i], dest[i]);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Div, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.mat[i], dest[i]);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::DivLeft, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.mat[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Add, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], exp.mat[i], dest[i] * exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Sub, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], exp.mat[i], -dest[i] * exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::SubLeft, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], -exp.mat[i], dest[i] * exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Mul, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Mul>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.mat);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Div, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Div>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.mat);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::DivLeft, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::DivLeft>& exp) {
		div(dest, exp.mat);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Add, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.mat[i] + exp.scalar;
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Sub, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.mat[i] - exp.scalar;
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::SubLeft, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.scalar - exp.mat[i];
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Mul, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Mul>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Div, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Div>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::DivLeft, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::DivLeft>& exp) {
		mul(dest, exp.mat);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Add, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Add>& exp) {
		dest = exp.mat;
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Sub, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Sub>& exp) {
		dest = exp.mat;
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::SubLeft, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::SubLeft>& exp) {
		dest = exp.mat;
		mul(dest, -1.0);
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.mat.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Mul, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Mul>& exp) {
		dest = exp.mat;
		mul(dest, exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::Div, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::Div>& exp) {
		dest = exp.mat;
		div(dest, exp.scalar);
	};
};

template <size_t ra, size_t ca>
struct SMOPImpl<float, ra, ca, OPType::DivLeft, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const SMOP<float, ra, ca, OPType::DivLeft>& exp) {
		dest.setConstant(exp.scalar);
		div(dest, exp.mat);
	};
};

// WHERE RESULT IS ADDED TO DEST

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Add, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Add>& exp) {
		cblas_saxpy(exp.lhs.size(), 1.0, exp.lhs(), 1, dest(), 1);
		cblas_saxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Sub, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Sub>& exp) {
		cblas_saxpy(exp.lhs.size(), 1.0, exp.lhs(), 1, dest(), 1);
		cblas_saxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Mul, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Div, OPType::Add> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], 1.0 / exp.rhs[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Add, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Add>& exp) {
		cblas_saxpy(exp.lhs.size(), -1.0, exp.lhs(), 1, dest(), 1);
		cblas_saxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Sub, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Sub>& exp) {
		cblas_saxpy(exp.lhs.size(), -1.0, exp.lhs(), 1, dest(), 1);
		cblas_saxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Mul, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Div, OPType::Sub> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], 1.0 / exp.rhs[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Add, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], dest[i] * exp.rhs[i]);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Sub, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], -dest[i] * exp.rhs[i]);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Mul, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Mul>& exp) {
		mul(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Div, OPType::Mul> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Div>& exp) {
		div(dest, exp.rhs);
		mul(dest, exp.lhs);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Add, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] + exp.rhs[i];
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Sub, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] - exp.rhs[i];
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Mul, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Mul>& exp) {
		div(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Div, OPType::Div> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Div>& exp) {
		mul(dest, exp.rhs);
		div(dest, exp.lhs);
	};
};

// WHERE RESULT IS ASSIGNED TO DEST
template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Add, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Add>& exp) {
		dest = exp.lhs;
		cblas_saxpy(exp.rhs.size(), 1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Sub, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Sub>& exp) {
		dest = exp.lhs;
		cblas_saxpy(exp.rhs.size(), -1.0, exp.rhs(), 1, dest(), 1);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Mul, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Mul>& exp) {
		dest = exp.lhs;
		mul(dest, exp.rhs);
	};
};

template <size_t ra, size_t ca, size_t rb, size_t cb>
struct MMOPImpl<float, ra, ca, rb, cb, OPType::Div, OPType::Assign> {
	template <size_t __rows, size_t __cols>
	static void apply(Matrix<__rows, __cols, float>& dest,
					  const MMOP<float, ra, ca, rb, cb, OPType::Div>& exp) {
		dest = exp.lhs;
		div(dest, exp.rhs);
	};
};

}  // namespace impl

#undef _LINALG_MATMAT_SIZECHECK
#undef _LINALG_SELFMAT_SIZECHECK
#undef _LINALG_VECVEC_SIZECHECK
#undef _LINALG_SELFVEC_SIZECHECK

}  // namespace swnumeric
