#pragma once

#include <cstdint>
#include <vector>

#include "Libraries/Tensor/Tensor.hpp"
#include "Libraries/Tensor/TensorMath.hpp"

namespace swnumeric {

struct CircleGeometry {
	Vector3 center;
	double radius;

	bool pointInCircle(const Vector3& v) {
		return norm2Sq(v - center) < radius * radius;
	}

	bool pointInCircle(uint64_t u, const std::vector<Vector3>& points) {
		return pointInCircle(points[u]);
	}
};

}  // namespace swnumeric
