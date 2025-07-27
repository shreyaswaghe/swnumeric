#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "Libraries/Tensor/Tensor.hpp"
#include "Libraries/Tensor/TensorMath.hpp"

namespace swnumeric {

struct Edge {
	uint64_t u;
	uint64_t v;

	inline void flip() { std::swap(u, v); }

	inline bool isOrdered() const { return u < v; }

	inline Edge ordered() const {
		return Edge{.u = std::min(u, v), .v = std::max(u, v)};
	}

	inline bool isTopologicallyEquivalent(const Edge& other) const {
		return (u == other.u && v == other.v) || (u == other.v && v == other.u);
	}

	inline bool sharesVertexWith(const Edge& other) const {
		return u == other.u || u == other.v || v == other.u || v == other.v;
	}

	inline double length(const std::vector<Vector3>& points) const {
		return norm2(UtoV(points));
	}

	inline Vector3 UtoV(const std::vector<Vector3>& points) const {
		return points[v] - points[u];
	}
};

inline bool operator==(const Edge& e0, const Edge& e1) noexcept {
	return (e0.u == e1.u && e0.v == e1.v);
}

inline bool operator<(const Edge& e0, const Edge& e1) noexcept {
	return e0.u < e1.u || e0.v < e1.v;
}

inline bool operator<=(const Edge& e0, const Edge& e1) noexcept {
	return e0 < e1 || e0 == e1;
}

}  // namespace swnumeric
