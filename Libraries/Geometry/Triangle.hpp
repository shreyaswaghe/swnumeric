#pragma once

#include <vector>

#include "Edge.hpp"
#include "Libraries/Tensor/Tensor.hpp"

namespace swnumeric {

struct Triangle {
	uint64_t v0, v1, v2;

	//
	// Topological Operations
	//
	inline std::array<Edge, 3> edgeList() const {
		return {Edge{v0, v1}, Edge{v1, v2}, Edge{v2, v0}};
	}

	inline std::array<Edge, 3> orderedEdgeList() const {
		return {						 //
				Edge{v0, v1}.ordered(),	 //
				Edge{v1, v2}.ordered(),	 //
				Edge{v2, v0}.ordered()};
	}

	inline void reverseOrientation() {
		uint64_t temp = v0;
		v0 = v1;
		v1 = temp;
	}

	//
	// Geometric Operations
	//
	inline Vector3 getNormal(const std::vector<Vector3>& points) const {
		Vector3 vv0 = points[v0];
		Vector3 vv1 = points[v1];
		Vector3 vv2 = points[v2];

		vv1 -= vv0;
		vv2 -= vv0;
		crossTo(vv0, vv1, vv2);

		return vv0;
	}

	inline Vector3 getCentroid(const std::vector<Vector3>& points) const {
		Vector3 v = points[v0];
		v += points[v1];
		v += points[v2];
		return v / 3.0;
	}

	inline Vector3 getEdgeLengths(const std::vector<Vector3>& points) const {
		Vector3 l;
		l[0] = norm2(points[v0] - points[v1]);
		l[1] = norm2(points[v1] - points[v2]);
		l[2] = norm2(points[v2] - points[v0]);
		return l;
	}

	inline double area(const std::vector<Vector3>& points) const {
		// half mag of cross product bw vectors
		return 0.5 * norm2(getNormal(points));
	}

	inline static double areaFromLengths(double a, double b, double c) {
		double s = (a + b + c) / 2;
		return std::sqrt(s * (s - a) * (s - b) * (s - c));
	}
};

}  // namespace swnumeric
