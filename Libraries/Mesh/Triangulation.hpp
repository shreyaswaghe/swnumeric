#include <string>

#include "Libraries/DataStructs/SortableVector.hpp"
#include "Libraries/Geometry/Curve.hpp"
#include "Libraries/Geometry/Edge.hpp"
#include "Libraries/Geometry/Triangle.hpp"
#include "Libraries/Logging/plog/Log.h"

namespace swnumeric {

struct Triangulation {
	std::vector<Triangle> triangles;
	std::map<uint8_t, std::vector<uint32_t>> grpIdToTriIdx;
	std::vector<Curve> boundingCurves;

	inline uint32_t getNumTriangles() const { return triangles.size(); }

	inline uint32_t getNumGroups() const { return grpIdToTriIdx.size(); }

	inline std::vector<uint32_t>& getTrisInGroup(uint8_t grp) {
		return grpIdToTriIdx[grp];
	}

	inline SortableVector<Edge> getOrderedEdgeList() const {
		SortableVector<Edge> edges;
		for (const Triangle& tri : triangles) {
			const std::array<Edge, 3> triedges = tri.orderedEdgeList();
			edges.v.insert(edges.v.end(), triedges.begin(), triedges.end());
		}
		for (const Curve& curve : boundingCurves) {
			for (const Edge& e : curve.edges) {
				edges.v.push_back(e.ordered());
			}
		}
		return edges;
	}
};

inline bool isTriangulationTopoSealed(const Triangulation& triangulation) {
	std::map<Edge, int> edgeToSignSum;

	for (const Triangle& tri : triangulation.triangles) {
		for (const Edge& edge : tri.edgeList()) {
			Edge orderedEdge = edge.ordered();
			if (not edgeToSignSum.contains(orderedEdge)) {
				edgeToSignSum[orderedEdge] = 0;
			}
			edgeToSignSum[orderedEdge] += edge.isOrdered() ? 1 : -1;
		}
	}

	for (const Curve& curve : triangulation.boundingCurves) {
		for (const Edge& edge : curve.edges) {
			Edge e = {edge.u, edge.v};
			if (not edgeToSignSum.contains(e)) {
				edgeToSignSum[e] = 0;
			}
			edgeToSignSum[e] += e.isOrdered() ? 1 : -1;
		}
	}

	bool c0cont = true;
	constexpr uint8_t edgePrintCount = 10;
	uint8_t edgeCount = 0;
	for (const auto [k, v] : edgeToSignSum) {
		if (v != 0) {
			c0cont = false;

			LOGD_IF(edgeCount++ < edgePrintCount)
				<< "EDGE (" + std::to_string(k.u) + ", " + std::to_string(k.v) +
					   ") has sum sign: " + std::to_string(v);
		}
	}

	if (not c0cont) {
		LOGE << "TRIANGULATION not topologically sealed !!!";
	}
	return c0cont;
}

struct TriangulationConnectivity {
	std::map<uint32_t, uint32_t[2]> edgeToTriLeftAndRight;
	std::vector<Edge> edges;

	TriangulationConnectivity(const Triangulation& tri) {}
};

}  // namespace swnumeric
