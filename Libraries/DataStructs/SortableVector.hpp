#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace swnumeric {

template <typename E>
struct SortableVector {
	static_assert(noexcept(std::declval<E>() <= std::declval<E>()),
				  "Element type must be sortable!!!");

	std::vector<E> v;

	void sort();
	uint64_t find(const E& query);
	bool contains(const E& query);
};

template <typename E>
void SortableVector<E>::sort() {
	std::sort(v.begin(), v.end());
}

template <typename E>
uint64_t SortableVector<E>::find(const E& query) {
	uint64_t lo = 0, hi = v.size() - 1;

	while (lo <= hi) {
		uint64_t mid = (lo + hi) / 2;
		E midE = v[mid];

		if (midE == query)
			return mid;
		else if (midE < query)
			lo = mid + 1;
		else
			hi = mid - 1;
	}
	return static_cast<uint64_t>(-1);
}

template <typename E>
bool SortableVector<E>::contains(const E& query) {
	return find(query) != static_cast<uint64_t>(-1);
}

}  // namespace swnumeric
