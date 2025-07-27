#pragma once

#include <map>
#include <vector>

#include "Edge.hpp"
#include "Libraries/Tensor/Tensor.hpp"
#include "Libraries/Tensor/TensorMath.hpp"

namespace swnumeric {

struct Curve {
	std::vector<Edge> edges;
	enum class Orientation { X, O };

	inline bool isClosed() {
		std::map<uint64_t, uint8_t> signSum;
		for (const Edge& edge : edges) {
			signSum[edge.u] += 1;
			signSum[edge.v] -= 1;
		}
		for (const auto [k, v] : signSum) {
			if (v != 0) {
				return false;
			}
		}
		return true;
	}

	inline Curve::Orientation orientation(const Vector3& planeNormal,
										  const std::vector<Vector3>& points) {
		const Edge& e0 = edges[0];
		const Edge& e1 = edges[1];

		Vector3 v1 = points[e0.v];
		v1 -= points[e0.v];

		Vector3 v2 = points[e1.v];
		v2 -= points[e1.u];

		// 2d cross product
		Vector3 v1xv2 = cross(v1, v2);
		double z = dot(planeNormal, v1xv2);

		// x when inside the plane
		return z < 0 ? Curve::Orientation::X : Curve::Orientation::O;
	}
};

}  // namespace swnumeric
