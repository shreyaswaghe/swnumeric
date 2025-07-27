#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// your blas lib

#include "armpl.h"

namespace swnumeric {

class TensorLike;

template <typename T, size_t... dims>
class Tensor;

template <size_t sz = 0, typename T = double>
class Vector;

template <size_t rw = 0, size_t cl = 0, typename T = double>
class Matrix;

template <size_t sz, typename T>
void printVector(const Vector<sz, T>& x) {
	for (size_t i = 0; i < x.size(); i++) {
		std::cout << x[i] << " , ";
	}
	std::cout << std::endl;
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// CONVENIENT ALIASES
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
using Vector1 = Vector<1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector4 = Vector<4>;
using Vector5 = Vector<5>;
using Vector6 = Vector<6>;

using Matrix11 = Matrix<1, 1>;
using Matrix12 = Matrix<1, 2>;
using Matrix13 = Matrix<1, 3>;
using Matrix14 = Matrix<1, 4>;
using Matrix15 = Matrix<1, 5>;
using Matrix16 = Matrix<1, 6>;

using Matrix21 = Matrix<2, 1>;
using Matrix22 = Matrix<2, 2>;
using Matrix23 = Matrix<2, 3>;
using Matrix24 = Matrix<2, 4>;
using Matrix25 = Matrix<2, 5>;
using Matrix26 = Matrix<2, 6>;

using Matrix31 = Matrix<3, 1>;
using Matrix32 = Matrix<3, 2>;
using Matrix33 = Matrix<3, 3>;
using Matrix34 = Matrix<3, 4>;
using Matrix35 = Matrix<3, 5>;
using Matrix36 = Matrix<3, 6>;

using Matrix41 = Matrix<4, 1>;
using Matrix42 = Matrix<4, 2>;
using Matrix43 = Matrix<4, 3>;
using Matrix44 = Matrix<4, 4>;
using Matrix45 = Matrix<4, 5>;
using Matrix46 = Matrix<4, 6>;

using Matrix51 = Matrix<5, 1>;
using Matrix52 = Matrix<5, 2>;
using Matrix53 = Matrix<5, 3>;
using Matrix54 = Matrix<5, 4>;
using Matrix55 = Matrix<5, 5>;
using Matrix56 = Matrix<5, 6>;

namespace impl {

#define ALLOW_TTOP 0

constexpr uint8_t alignment = 16;

enum class OPType { Add, Sub, SubLeft, Mul, Div, DivLeft, Assign };

template <typename T, OPType _op, size_t... sa>
struct STOP {
	const Tensor<T, sa...>& vec;
	const T scalar;
	static constexpr OPType op = _op;
};

template <typename T, OPType _op, typename SeqA, typename SeqB>
struct TTOP;

template <typename T, OPType _op, size_t... sa, size_t... sb>
struct TTOP<T, _op, std::index_sequence<sa...>, std::index_sequence<sb...>> {
	const Tensor<T, sa...>& lhs;
	const Tensor<T, sb...>& rhs;
	static constexpr OPType op = _op;
};

template <typename T, OPType _op, OPType _destop, size_t... sa>
struct STOPImpl;

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

template <typename T, size_t _nDims1, size_t _nDims2>
void checkShape(const std::array<T, _nDims1>& a1,
				const std::array<T, _nDims2>& a2) {
	if (_nDims1 != _nDims2) {
		throw std::runtime_error("DIMENSION ERROR: " + std::to_string(_nDims1) +
								 " AND " + std::to_string(_nDims2));
	}
	for (size_t i = 0; i < _nDims1; i++) {
		if (a1[i] != a2[i]) {
			std::string err1 = "SIZE MISMATCH ERROR: (";
			for (size_t i = 0; i < _nDims1 - 1; i++) {
				err1 += std::to_string(a1[i]) + ", ";
			}
			err1 += std::to_string(a1[_nDims1 - 1]) + ") AND ";
			for (size_t i = 0; i < _nDims1 - 1; i++) {
				err1 += std::to_string(a2[i]) + ", ";
			}
			err1 += std::to_string(a2[_nDims1 - 1]) + ") !!!";
			throw std::runtime_error(err1);
		}
	}
};

class TensorLike {};

template <typename T, size_t... dims>
class Tensor : public TensorLike {
   protected:
	static constexpr size_t _ctSize = ([]() {
		size_t s = 1;
		for (const size_t d : std::array{dims...}) s *= d;
		return s;
	})();
	static constexpr std::array<uint32_t, sizeof...(dims)> compileTimeShape = {
		dims...};
	static constexpr size_t _nDims = sizeof...(dims);

	union {
		T* heapData;
		T staticData[_ctSize > 0 ? _ctSize : 1];
	} dataHolder;

	T* data = nullptr;
	size_t _size = 0;
	std::array<uint32_t, _nDims> _shape;

   public:
	static consteval size_t _compileTimeSize(
		const std::array<uint32_t, _nDims>& arr) {
		size_t sz = 1;
		for (size_t i = 0; i < _nDims; i++) sz *= arr[i];
		return sz;
	}

	static constexpr int _stride = 1;
	using value_type = T;

	void alloc(const std::array<uint32_t, _nDims>& shape) {
		if (isAlloced()) return;

		size_t s = 1;
		for (uint32_t i : shape) {
			s *= i;
		}

		if constexpr (_ctSize > 0) {
			_shape = {dims...};
			_size = _ctSize;
			data = dataHolder.staticData;
		} else {
			_shape = shape;
			_size = s;
			size_t allocedBytes =
				((sizeof(T) * s + impl::alignment - 1) / impl::alignment) *
				impl::alignment;

			void* ptr = std::aligned_alloc(impl::alignment, allocedBytes);
			if (!ptr) throw std::bad_alloc();

			dataHolder.heapData = static_cast<T*>(ptr);
			data = dataHolder.heapData;
		}
		std::fill_n(data, _size, T(0.0));
	}

	// shape descriptors
	inline size_t size() const { return _size; }
	inline size_t comptimeSize() const { return _ctSize; }
	inline bool isStaticSized() const { return _ctSize > 0; }
	inline bool isAlloced() const { return static_cast<bool>(data); }

	inline uint8_t nDims() const { return _nDims; }
	inline std::array<uint32_t, _nDims> shape() const { return _shape; }
	inline int stride() const { return _stride; }

	// accessors
	T* ptr() { return data; }
	const T* ptr() const { return data; }

	inline size_t idx(std::array<size_t, _nDims>& iidx) const {
		size_t offset = 0;
		size_t stride = 1;
		for (uint8_t i = 0; i < _nDims; i++) {
			offset += iidx[i] * stride;
			stride *= _shape[i];
		}
		return offset;
	}

	T* operator()() { return ptr(); }
	const T* operator()() const { return ptr(); }
	T* operator()(const std::array<size_t, _nDims>& iidx) {
		return ptr() + idx(iidx);
	}
	const T* operator()(std::array<size_t, _nDims>& iidx) const {
		return ptr() + idx(iidx);
	}
	T& operator[](size_t iidx) { return ptr()[iidx]; };
	const T& operator[](size_t iidx) const { return ptr()[iidx]; };

	// simple setters
	void setOne() { std::fill_n(ptr(), size(), T(1.0)); }
	void setZero() { std::fill_n(ptr(), size(), T(0.0)); }
	void setConstant(const T val) { std::fill_n(ptr(), size(), T(val)); }

	// copy on assignment
	template <size_t... ddims>
	Tensor<T, dims...>& operator=(const Tensor<T, ddims...>& other) {
		checkShape(shape(), other.shape());
		std::memcpy(ptr(), other.ptr(), sizeof(T) * other.size());
		return *this;
	}

	Tensor<T, dims...>& operator=(const Tensor<T, dims...>& other) {
		if (isAlloced())
			checkShape(shape(), other.shape());
		else
			alloc(other.shape());

		std::memcpy(ptr(), other.ptr(), sizeof(T) * other.size());
		return *this;
	}

	// constructor
	Tensor(const std::array<uint32_t, _nDims> sz) { alloc(sz); }

	Tensor() {
		if constexpr (_ctSize > 0) alloc({});
	}

	Tensor(const Tensor<T, dims...>& other) : Tensor() {
		alloc(other.shape());
		(*this) = other;
	};

	Tensor(Tensor<T, dims...>&& other) noexcept
		: data(other.data), _size(other._size), _shape(other._shape) {
		// Need to handle the union properly
		if constexpr (_ctSize > 0) {
			// For static data, we need to copy the data, not move pointers
			data = dataHolder.staticData;
			std::memcpy(dataHolder.staticData, other.dataHolder.staticData,
						sizeof(T) * _ctSize);
		} else {
			// For heap data, we can move the pointer
			dataHolder.heapData = other.dataHolder.heapData;
			other.dataHolder.heapData = nullptr;
		}

		other.data = nullptr;
		other._size = 0;
		other._shape = {};
	}

	// Move assignment operator
	Tensor<T, dims...>& operator=(Tensor<T, dims...>&& other) noexcept {
		if (this == &other) return *this;

		// Free current resources
		free();

		// Move data
		_size = other._size;
		_shape = other._shape;

		if constexpr (_ctSize > 0) {
			data = dataHolder.staticData;
			std::memcpy(dataHolder.staticData, other.dataHolder.staticData,
						sizeof(T) * _ctSize);
		} else {
			dataHolder.heapData = other.dataHolder.heapData;
			data = dataHolder.heapData;
			other.dataHolder.heapData = nullptr;
		}

		other.data = nullptr;
		other._size = 0;
		other._shape = {};

		return *this;
	}

	// destructor
	inline void free() {
		if constexpr (_ctSize > 0) {
		} else {
			std::free(dataHolder.heapData);
		}
		data = nullptr;
		_size = 0;
		_shape = {};
	}

	~Tensor() { free(); }

	// expression enablement
	template <size_t... sa, impl::OPType _op>
	Tensor<T, dims...>& operator=(const impl::STOP<T, _op, sa...>& exp);
	template <impl::OPType _op>
	Tensor<T, dims...>& operator=(const impl::STOP<T, _op, dims...>& exp);

#if ALLOW_TTOP
	template <size_t... sa, size_t... sb, impl::OPType _op>
	Tensor<T, dims...>& operator=(
		const impl::TTOP<T, _op, std::index_sequence<sa...>,
						 std::index_sequence<sb...>>& exp);
#endif

	template <size_t... sa>
	Tensor<T, dims...>& operator+=(const Tensor<T, sa...>& a);
	Tensor<T, dims...>& operator+=(const T& a);
	template <size_t... sa, impl::OPType _op>
	Tensor<T, dims...>& operator+=(const impl::STOP<T, _op, sa...>& a);

#if ALLOW_TTOP
	template <size_t... sa, size_t... sb, impl::OPType _op>
	Tensor<T, dims...>& operator+=(
		const impl::TTOP<T, _op, std::index_sequence<sa...>,
						 std::index_sequence<sb...>>& a);
#endif

	template <size_t... sa>
	Tensor<T, dims...>& operator-=(const Tensor<T, sa...>& a);
	Tensor<T, dims...>& operator-=(const T& a);
	template <size_t... sa, impl::OPType _op>
	Tensor<T, dims...>& operator-=(const impl::STOP<T, _op, sa...>& a);

#if ALLOW_TTOP
	template <size_t... sa, size_t... sb, impl::OPType _op>
	Tensor<T, dims...>& operator-=(
		const impl::TTOP<T, _op, std::index_sequence<sa...>,
						 std::index_sequence<sb...>>& a);
#endif

	template <size_t... sa>
	Tensor<T, dims...>& operator*=(const Tensor<T, sa...>& a);
	Tensor<T, dims...>& operator*=(const T& a);
	template <size_t... sa, impl::OPType _op>
	Tensor<T, dims...>& operator*=(const impl::STOP<T, _op, sa...>& a);

#if ALLOW_TTOP
	template <size_t... sa, size_t... sb, impl::OPType _op>
	Tensor<T, dims...>& operator*=(
		const impl::TTOP<T, _op, std::index_sequence<sa...>,
						 std::index_sequence<sb...>>& a);
#endif

	template <size_t... sa>
	Tensor<T, dims...>& operator/=(const Tensor<T, sa...>& a);
	Tensor<T, dims...>& operator/=(const T& a);
	template <size_t... sa, impl::OPType _op>
	Tensor<T, dims...>& operator/=(const impl::STOP<T, _op, sa...>& a);

#if ALLOW_TTOP
	template <size_t... sa, size_t... sb, impl::OPType _op>
	Tensor<T, dims...>& operator/=(
		const impl::TTOP<T, _op, std::index_sequence<sa...>,
						 std::index_sequence<sb...>>& a);
#endif
};

template <size_t sz, typename T>
class Vector : public Tensor<T, sz> {
   public:
	using Tensor<T, sz>::operator=;
	using Tensor<T, sz>::operator+=;
	using Tensor<T, sz>::operator-=;
	using Tensor<T, sz>::operator*=;
	using Tensor<T, sz>::operator/=;

	inline size_t idx(std::array<size_t, 1>& iidx) const { return iidx[0]; }
	inline size_t idx(size_t iidx) const { return iidx; }
	T* operator()(size_t iidx) { return this->ptr() + iidx; }
	const T* operator()(size_t iidx) const { return this->ptr() + iidx; }

	Vector() : Tensor<T, sz>() {}
	Vector(uint32_t size) : Tensor<T, sz>({size}) {}
	template <size_t ssz>
	Vector(const Vector<ssz, T>& x) : Tensor<T, sz>(x) {}

	Vector(const std::initializer_list<T>& arr)
		: Tensor<T, sz>({static_cast<uint32_t>(arr.size())}) {
		static_assert(sz > 0,
					  "CAN ONLY SET STATIC SIZED VECTOR WITH INIT LIST.");
		std::memcpy(this->ptr(), arr.begin(), sizeof(T) * sz);
	}

	template <size_t ssz, impl::OPType _op>
	Vector<sz, T>& operator=(const impl::STOP<T, _op, ssz>& exp) {
		Tensor<T, sz>::operator=(exp);
		return *this;
	}

	Matrix<1, sz, T> asRowMatrix() const {
		Matrix<1, sz, T> x(1, this->size());
		std::memcpy(x.ptr(), this->ptr(), this->size() * sizeof(T));
		return x;
	}

	Matrix<sz, sz, T> asDiagonalMatrix() const {
		Matrix<sz, sz, T> x(this->size(), this->size());
		for (size_t i = 0; i < this->size(); i++) {
			x[i * this->size() + i] = this->ptr()[i];
		}
		return (x);
	}
};

template <size_t rw, size_t cl, typename T>
class Matrix : public Tensor<T, rw, cl> {
	size_t _rows, _cols;

   public:
	using Tensor<T, rw, cl>::operator=;
	using Tensor<T, rw, cl>::operator+=;
	using Tensor<T, rw, cl>::operator-=;
	using Tensor<T, rw, cl>::operator*=;
	using Tensor<T, rw, cl>::operator/=;

	inline size_t idx(std::array<size_t, 2>& iidx) const {
		return iidx[0] + rows() * iidx[1];
	}
	inline size_t idx(size_t iidx0, size_t iidx1) const {
		return iidx0 + rows() * iidx1;
	}
	T* operator()(size_t iidx0, size_t iidx1) {
		return this->ptr() + iidx0 + rows() * iidx1;
	}
	const T* operator()(size_t iidx0, size_t iidx1) const {
		return this->ptr() + iidx0 + rows() * iidx1;
	}

	Matrix() : Tensor<T, rw, cl>(), _rows(rw), _cols(cl) {};

	Matrix(uint32_t rows, uint32_t cols)
		: Tensor<T, rw, cl>({rows, cols}), _rows(rows), _cols(cols) {}

	template <size_t rrw, size_t ccl>
	Matrix(const Matrix<rrw, ccl, T>& other)
		: Tensor<T, rw, cl>(other.shape()),
		  _rows(other.rows()),
		  _cols(other.cols()) {}

	inline size_t rows() const { return _rows; }
	inline size_t lda() const { return _rows; }
	inline size_t cols() const { return _cols; }

	Vector<0, T> diagonalAsVector() const {
		size_t m = std::min(rows(), cols());
		Vector<0, T> x(m);
		for (size_t i = 0; i < m; i++) {
			x[i] = *((*this)(i, i));
		}
		return (x);
	}

	Vector<rw, T> col(uint32_t i) {
		uint32_t c = i;
		Vector<rw, T> x(rows());
		for (uint32_t j = 0; j < rows(); j++) {
			x[j] = *(*this)(j, c);
		}
		return (x);
	}

	Vector<cl, T> row(uint32_t i) {
		uint32_t r = i;
		Vector<cl, T> x(cols());
		for (uint32_t j = 0; j < cols(); j++) {
			x[j] = *(*this)(r, j);
		}
		return (x);
	}
};

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///
/// OPERATORS
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

template <typename T, size_t... sa>
auto operator+(const Tensor<T, sa...>& a) {
	return a;
}

template <typename T, size_t... sa>
auto operator-(const Tensor<T, sa...>& a) {
	return T(-1.0) * a;
}

template <typename T, size_t... sa>
auto operator+(const Tensor<T, sa...>& a, const T scalar) {
	return impl::STOP<T, impl::OPType::Add, sa...>{.vec = a, .scalar = scalar};
}

template <typename T, size_t... sa>
auto operator+(const T scalar, const Tensor<T, sa...>& a) {
	return impl::STOP<T, impl::OPType::Add, sa...>{.vec = a, .scalar = scalar};
}

template <typename T, size_t... sa>
auto operator-(const Tensor<T, sa...>& a, const T scalar) {
	return impl::STOP<T, impl::OPType::Sub, sa...>{.vec = a, .scalar = scalar};
}

template <typename T, size_t... sa>
auto operator-(const T scalar, const Tensor<T, sa...>& a) {
	return impl::STOP<T, impl::OPType::SubLeft, sa...>{.vec = a,
													   .scalar = scalar};
}

template <typename T, size_t... sa>
auto operator*(const Tensor<T, sa...>& a, T scalar) {
	return impl::STOP<T, impl::OPType::Mul, sa...>{.vec = a, .scalar = scalar};
}

template <typename T, size_t... sa>
auto operator*(const T scalar, const Tensor<T, sa...>& a) {
	return impl::STOP<T, impl::OPType::Mul, sa...>{.vec = a, .scalar = scalar};
}

template <typename T, size_t... sa>
auto operator/(const Tensor<T, sa...>& a, const T scalar) {
	return impl::STOP<T, impl::OPType::Div, sa...>{.vec = a, .scalar = scalar};
}

template <typename T, size_t... sa>
auto operator/(const T scalar, const Tensor<T, sa...>& a) {
	return impl::STOP<T, impl::OPType::DivLeft, sa...>{.vec = a,
													   .scalar = scalar};
}

#if ALLOW_TTOP

template <typename T, size_t... sa, size_t... sb>
auto operator+(const Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
	return impl::TTOP<T, impl::OPType::Add, std::index_sequence<sa...>,
					  std::index_sequence<sb...>>{.lhs = a, .rhs = b};
}

template <typename T, size_t... sa, size_t... sb>
auto operator-(const Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
	return impl::TTOP<T, impl::OPType::Sub, std::index_sequence<sa...>,
					  std::index_sequence<sb...>>{.lhs = a, .rhs = b};
}

template <typename T, size_t... sa, size_t... sb>
auto operator*(const Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
	return impl::TTOP<T, impl::OPType::Mul, std::index_sequence<sa...>,
					  std::index_sequence<sb...>>{.lhs = a, .rhs = b};
}

template <typename T, size_t... sa, size_t... sb>
auto operator/(const Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
	return impl::TTOP<T, impl::OPType::Div, std::index_sequence<sa...>,
					  std::index_sequence<sb...>>{.lhs = a, .rhs = b};
}

#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
///
/// NOBLAS ATOMIC OPERATIONS FOR Tensor
///
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template <typename T, size_t... sa, size_t... sb>
void add(Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
}

template <typename T, size_t... sa>
void add(Tensor<T, sa...>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] += b;
}

template <typename T, size_t... sa, size_t... sb>
void sub(Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] -= b[i];
}

template <typename T, size_t... sa>
void sub(Tensor<T, sa...>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] -= b;
}

template <typename T, size_t... sa, size_t... sb>
void subLeft(Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b[i] - a[i];
}

template <typename T, size_t... sa>
void subLeft(Tensor<T, sa...>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b - a[i];
}

template <typename T, size_t... sa, size_t... sb>
void mul(Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] *= b[i];
}

template <typename T, size_t... sa>
void mul(Tensor<T, sa...>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] *= b;
}

template <typename T, size_t... sa, size_t... sb>
void div(Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] /= b[i];
}

template <typename T, size_t... sa>
void div(Tensor<T, sa...>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] /= b;
}

template <typename T, size_t... sa, size_t... sb>
void divLeft(Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b[i] / a[i];
}

template <typename T, size_t... sa>
void divLeft(Tensor<T, sa...>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (size_t i = 0; i < a.size(); i++) a[i] = b / a[i];
}

template <typename T, size_t... sa, size_t... sb>
void copyTo(Tensor<T, sa...>& a, const Tensor<T, sb...>& b) {
	std::memcpy(a(), b(), a.size() * sizeof(T));
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///
///     COMPUTATIONAL KERNEL DEFINITIONS
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

namespace impl {

// WHERE RESULT IS ADDED TO DEST

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Add, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Add, sb...>& exp) {
		add(dest, exp.scalar);
		add(dest, exp.vec);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Sub, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Sub, sb...>& exp) {
		add(dest, exp.vec);
		sub(dest, exp.scalar);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::SubLeft, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::SubLeft, sb...>& exp) {
		sub(dest, exp.vec);
		add(dest, exp.scalar);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Mul, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Mul, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Div, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Div, sb...>& exp) {
		T sc = 1.0 / exp.scalar;
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(sc, exp.vec[i], dest[i]);
	};
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::DivLeft, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::DivLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Add, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Add, sb...>& exp) {
		sub(dest, exp.scalar);
		sub(dest, exp.vec);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Sub, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Sub, sb...>& exp) {
		sub(dest, exp.vec);
		add(dest, exp.scalar);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::SubLeft, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::SubLeft, sb...>& exp) {
		add(dest, exp.vec);
		sub(dest, exp.scalar);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Mul, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Mul, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Div, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Div, sb...>& exp) {
		T sc = -1.0 / exp.scalar;
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(sc, exp.vec[i], dest[i]);
	};
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::DivLeft, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::DivLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Add, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Add, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], dest[i] * exp.scalar);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Sub, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Sub, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], -dest[i] * exp.scalar);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::SubLeft, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::SubLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], -exp.vec[i], dest[i] * exp.scalar);
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Mul, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Mul, sb...>& exp) {
		mul(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Div, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Div, sb...>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::DivLeft, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::DivLeft, sb...>& exp) {
		div(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

// WHERE RESULT IS DIVIDED BY DEST

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Add, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Add, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] + exp.scalar;
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Sub, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Sub, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] - exp.scalar;
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::SubLeft, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::SubLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.scalar - exp.vec[i];
	}
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Mul, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Mul, sb...>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::Div, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::Div, sb...>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <typename T, size_t... sa>
struct STOPImpl<T, OPType::DivLeft, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<T, sb...>& dest,
					  const STOP<T, OPType::DivLeft, sb...>& exp) {
		mul(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

//
// SPECIALIZATION FOR DOUBLE
//

// WHERE RESULT IS ADDED TO DEST
template <size_t... sa>
struct STOPImpl<double, OPType::Add, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Add, sa...>& exp) {
		cblas_daxpy(exp.vec.size(), 1.0, exp.vec(), 1, dest(), 1);

		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Sub, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Sub, sb...>& exp) {
		cblas_daxpy(dest.size(), 1.0, exp.vec(), 1, dest(), 1);

		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::SubLeft, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::SubLeft, sb...>& exp) {
		cblas_daxpy(dest.size(), -1.0, exp.vec(), 1, dest(), 1);

		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Mul, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Mul, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Div, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Div, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::DivLeft, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::DivLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t... sa>
struct STOPImpl<double, OPType::Add, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Add, sb...>& exp) {
		cblas_daxpy(exp.vec.size(), -1.0, exp.vec(), 1, dest(), 1);
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Sub, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Sub, sb...>& exp) {
		cblas_daxpy(exp.vec.size(), -1.0, exp.vec(), 1, dest(), 1);
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::SubLeft, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::SubLeft, sb...>& exp) {
		cblas_daxpy(exp.vec.size(), 1.0, exp.vec(), 1, dest(), 1);
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Mul, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Mul, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Div, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Div, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::DivLeft, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::DivLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t... sa>
struct STOPImpl<double, OPType::Add, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Add, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], dest[i] * exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Sub, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Sub, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], -dest[i] * exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::SubLeft, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::SubLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], -exp.vec[i], dest[i] * exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Mul, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Mul, sb...>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Div, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Div, sb...>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::DivLeft, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::DivLeft, sb...>& exp) {
		div(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t... sa>
struct STOPImpl<double, OPType::Add, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Add, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] + exp.scalar;
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Sub, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Sub, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] - exp.scalar;
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::SubLeft, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::SubLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.scalar - exp.vec[i];
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Mul, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Mul, sb...>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Div, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Div, sb...>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::DivLeft, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::DivLeft, sb...>& exp) {
		mul(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <size_t... sa>
struct STOPImpl<double, OPType::Add, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Add, sb...>& exp) {
		dest = exp.vec;
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Sub, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Sub, sb...>& exp) {
		dest = exp.vec;
		alignas(16) double x[1] = {-exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::SubLeft, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::SubLeft, sb...>& exp) {
		dest = exp.vec;
		mul(dest, -1.0);
		alignas(16) double x[1] = {exp.scalar};
		cblas_daxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Mul, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Mul, sb...>& exp) {
		dest = exp.vec;
		mul(dest, exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::Div, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::Div, sb...>& exp) {
		dest = exp.vec;
		div(dest, exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<double, OPType::DivLeft, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<double, sa...>& dest,
					  const STOP<double, OPType::DivLeft, sb...>& exp) {
		dest.setConstant(exp.scalar);
		div(dest, exp.vec);
	};
};

//
// SPECIALIZATION FOR FLOAT
//

// WHERE RESULT IS ADDED TO DEST
template <size_t... sa>
struct STOPImpl<float, OPType::Add, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Add, sa...>& exp) {
		cblas_saxpy(exp.vec.size(), 1.0, exp.vec(), 1, dest(), 1);

		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Sub, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Sub, sb...>& exp) {
		cblas_saxpy(dest.size(), 1.0, exp.vec(), 1, dest(), 1);

		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::SubLeft, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::SubLeft, sb...>& exp) {
		cblas_saxpy(dest.size(), -1.0, exp.vec(), 1, dest(), 1);

		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Mul, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Mul, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Div, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Div, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::DivLeft, OPType::Add, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::DivLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <size_t... sa>
struct STOPImpl<float, OPType::Add, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Add, sb...>& exp) {
		cblas_saxpy(exp.vec.size(), -1.0, exp.vec(), 1, dest(), 1);
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Sub, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Sub, sb...>& exp) {
		cblas_saxpy(exp.vec.size(), -1.0, exp.vec(), 1, dest(), 1);
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::SubLeft, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::SubLeft, sb...>& exp) {
		cblas_saxpy(exp.vec.size(), 1.0, exp.vec(), 1, dest(), 1);
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Mul, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Mul, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Div, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Div, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::DivLeft, OPType::Sub, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::DivLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <size_t... sa>
struct STOPImpl<float, OPType::Add, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Add, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], dest[i] * exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Sub, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Sub, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], -dest[i] * exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::SubLeft, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::SubLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], -exp.vec[i], dest[i] * exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Mul, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Mul, sb...>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Div, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Div, sb...>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::DivLeft, OPType::Mul, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::DivLeft, sb...>& exp) {
		div(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <size_t... sa>
struct STOPImpl<float, OPType::Add, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Add, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] + exp.scalar;
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Sub, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Sub, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] - exp.scalar;
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::SubLeft, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::SubLeft, sb...>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (size_t i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.scalar - exp.vec[i];
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Mul, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Mul, sb...>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Div, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Div, sb...>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::DivLeft, OPType::Div, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::DivLeft, sb...>& exp) {
		mul(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <size_t... sa>
struct STOPImpl<float, OPType::Add, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Add, sb...>& exp) {
		dest = exp.vec;
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Sub, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Sub, sb...>& exp) {
		dest = exp.vec;
		alignas(16) float x[1] = {-exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::SubLeft, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::SubLeft, sb...>& exp) {
		dest = exp.vec;
		mul(dest, -1.0);
		alignas(16) float x[1] = {exp.scalar};
		cblas_saxpy(exp.vec.size(), 1.0, x, 0, dest(), 1);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Mul, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Mul, sb...>& exp) {
		dest = exp.vec;
		mul(dest, exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::Div, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::Div, sb...>& exp) {
		dest = exp.vec;
		div(dest, exp.scalar);
	};
};

template <size_t... sa>
struct STOPImpl<float, OPType::DivLeft, OPType::Assign, sa...> {
	template <size_t... sb>
	static void apply(Tensor<float, sa...>& dest,
					  const STOP<float, OPType::DivLeft, sb...>& exp) {
		dest.setConstant(exp.scalar);
		div(dest, exp.vec);
	};
};
}  // namespace impl

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///
///    DISPATCH TO THE RIGHT KERNEL ON ASSIGNMENT
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

template <typename T, size_t... dims>
template <size_t... sa>
Tensor<T, dims...>& Tensor<T, dims...>::operator+=(const Tensor<T, sa...>& a) {
	checkShape(shape(), a.shape());
	if constexpr (std::is_same_v<T, double>) {
		cblas_daxpy(size(), 1.0, a.ptr(), a.stride(), ptr(), stride());
	} else if constexpr (std::is_same_v<T, float>) {
		cblas_saxpy(size(), 1.0, a.ptr(), a.stride(), ptr(), stride());
	} else {
		add(*this, a);
	}
	return *this;
}

template <typename T, size_t... dims>
Tensor<T, dims...>& Tensor<T, dims...>::operator+=(const T& a) {
	if constexpr (std::is_same_v<T, double>) {
		double aa[1] = {a};
		cblas_daxpy(size(), 1.0, aa, 0, ptr(), stride());
	} else if constexpr (std::is_same_v<T, float>) {
		float aa[1] = {a};
		cblas_saxpy(size(), 1.0, aa, 0, ptr(), stride());
	} else {
		add(*this, a);
	}
	return *this;
};

template <typename T, size_t... dims>
template <size_t... sa, impl::OPType _op>
Tensor<T, dims...>& Tensor<T, dims...>::operator+=(
	const impl::STOP<T, _op, sa...>& a) {
	checkShape(shape(), a.vec.shape());
	impl::STOPImpl<T, _op, impl::OPType::Add, sa...>::apply(*this, a);
	return *this;
}

template <typename T, size_t... dims>
template <size_t... sa>
Tensor<T, dims...>& Tensor<T, dims...>::operator-=(const Tensor<T, sa...>& a) {
	checkShape(shape(), a.shape());
	if constexpr (std::is_same_v<T, double>) {
		cblas_daxpy(size(), -1.0, a.ptr(), a.stride(), ptr(), stride());
	} else if constexpr (std::is_same_v<T, float>) {
		cblas_saxpy(size(), -1.0, a.ptr(), a.stride(), ptr(), stride());
	} else {
		add(*this, a);
	}
	return *this;
}

template <typename T, size_t... dims>
Tensor<T, dims...>& Tensor<T, dims...>::operator-=(const T& a) {
	if constexpr (std::is_same_v<T, double>) {
		double aa[1] = {a};
		cblas_daxpy(size(), -1.0, aa, 0, ptr(), stride());
	} else if constexpr (std::is_same_v<T, float>) {
		float aa[1] = {a};
		cblas_saxpy(size(), -1.0, aa, 0, ptr(), stride());
	} else {
		add(*this, a);
	}
	return *this;
};

template <typename T, size_t... dims>
template <size_t... sa, impl::OPType _op>
Tensor<T, dims...>& Tensor<T, dims...>::operator-=(
	const impl::STOP<T, _op, sa...>& a) {
	checkShape(shape(), a.vec.shape());
	impl::STOPImpl<T, _op, impl::OPType::Sub, sa...>::apply(*this, a);
	return *this;
}

template <typename T, size_t... dims>
template <size_t... sa>
Tensor<T, dims...>& Tensor<T, dims...>::operator*=(const Tensor<T, sa...>& a) {
	checkShape(shape(), a.shape());
	mul(*this, a);
	return *this;
}

template <typename T, size_t... dims>
Tensor<T, dims...>& Tensor<T, dims...>::operator*=(const T& a) {
	mul(*this, a);
	return *this;
};

template <typename T, size_t... dims>
template <size_t... sa, impl::OPType _op>
Tensor<T, dims...>& Tensor<T, dims...>::operator*=(
	const impl::STOP<T, _op, sa...>& a) {
	checkShape(shape(), a.vec.shape());
	impl::STOPImpl<T, _op, impl::OPType::Mul, sa...>::apply(*this, a);
	return *this;
}

template <typename T, size_t... dims>
template <size_t... sa>
Tensor<T, dims...>& Tensor<T, dims...>::operator/=(const Tensor<T, sa...>& a) {
	checkShape(shape(), a.shape());
	div(*this, a);
	return *this;
}

template <typename T, size_t... dims>
Tensor<T, dims...>& Tensor<T, dims...>::operator/=(const T& a) {
	div(*this, a);
	return *this;
};

template <typename T, size_t... dims>
template <size_t... sa, impl::OPType _op>
Tensor<T, dims...>& Tensor<T, dims...>::operator/=(
	const impl::STOP<T, _op, sa...>& a) {
	checkShape(shape(), a.vec.shape());
	impl::STOPImpl<T, _op, impl::OPType::Div, sa...>::apply(*this, a);
	return *this;
}

template <typename T, size_t... dims>
template <size_t... sa, impl::OPType _op>
Tensor<T, dims...>& Tensor<T, dims...>::operator=(
	const impl::STOP<T, _op, sa...>& exp) {
	checkShape(shape(), exp.vec.shape());
	impl::STOPImpl<T, _op, impl::OPType::Assign, sa...>::apply(*this, exp);
	return *this;
}
template <typename T, size_t... dims>
template <impl::OPType _op>
Tensor<T, dims...>& Tensor<T, dims...>::operator=(
	const impl::STOP<T, _op, dims...>& exp) {
	checkShape(shape(), exp.vec.shape());
	impl::STOPImpl<T, _op, impl::OPType::Assign, dims...>::apply(*this, exp);
	return *this;
}

template <typename T, size_t sz>
Vector<sz, T> operator+(const Vector<sz, T>& a, const Vector<sz, T>& b) {
	Vector<sz, T> c = a;
	c += b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator-(const Vector<sz, T>& a, const Vector<sz, T>& b) {
	Vector<sz, T> c = a;
	c -= b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator*(const Vector<sz, T>& a, const Vector<sz, T>& b) {
	Vector<sz, T> c = a;
	c *= b;
	return c;
}
template <typename T, size_t sz>
Vector<sz, T> operator/(const Vector<sz, T>& a, const Vector<sz, T>& b) {
	Vector<sz, T> c = a;
	c /= b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator+(const Vector<sz, T>& a, const T& b) {
	Vector<sz, T> c = a;
	c += b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator-(const Vector<sz, T>& a, const T& b) {
	Vector<sz, T> c = a;
	c -= b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator*(const Vector<sz, T>& a, const T& b) {
	Vector<sz, T> c = a;
	c *= b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator/(const Vector<sz, T>& a, const T& b) {
	Vector<sz, T> c = a;
	c /= b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator+(const T& b, const Vector<sz, T>& a) {
	Vector<sz, T> c = a;
	c += b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator-(const T& b, const Vector<sz, T>& a) {
	Vector<sz, T> c = a;
	c -= b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator*(const T& b, const Vector<sz, T>& a) {
	Vector<sz, T> c = a;
	c *= b;
	return c;
}

template <typename T, size_t sz>
Vector<sz, T> operator/(const T& b, const Vector<sz, T>& a) {
	Vector<sz, T> c = a;
	c /= b;
	return c;
}

}  // namespace swnumeric
