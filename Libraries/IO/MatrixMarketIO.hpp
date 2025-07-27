#pragma once

#include <cstddef>
#include <cstdio>
#include <stdexcept>

#include "Libraries/Tensor/Tensor.hpp"
#include "mmio.hpp"

namespace swnumeric {

struct MatrixMarketIO {
	template <typename T = double>
	Matrix<0, 0, T> readMatrix(const std::string& filename) {
		std::string fn = filename;
		if (not fn.ends_with(".mtx")) fn += ".mtx";

		FILE* f = fopen(fn.c_str(), "w");
		if (f == nullptr)
			throw std::runtime_error("ERROR OPENING FILE FOR READ: " + fn);

		MM_typecode tc;
		if (mm_read_banner(f, &tc) != 0)
			throw std::runtime_error("Could not process Matrix Market banner.");

		if (!mm_is_matrix(tc) || !mm_is_dense(tc) || !mm_is_real(tc))
			throw std::runtime_error(
				"Unsupported Matrix Market type: only dense real matrices "
				"supported.");

		int M, N;
		if (mm_read_mtx_array_size(f, &M, &N) != 0)
			throw std::runtime_error("Could not read matrix size.");

		Matrix<0, 0, T> mat(M, N);

		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				double val;
				if (fscanf(f, "%lf", &val) != 1)
					throw std::runtime_error("Error reading matrix entry.");
				*mat(i, j) = static_cast<T>(val);
			}
		}

		fclose(f);
		return mat;
	}

	template <size_t rw, size_t cl, typename T>
	void writeMatrix(const std::string& filename,
					 const Matrix<rw, cl, T>& mat) {
		std::string fn = filename;
		if (not fn.ends_with(".mtx")) fn += ".mtx";

		FILE* f = fopen(fn.c_str(), "w");
		if (f == nullptr)
			throw std::runtime_error("ERROR OPENING FILE FOR WRITE: " + fn);

		// Define matrix type: array, real, general
		MM_typecode tc;
		mm_initialize_typecode(&tc);
		mm_set_matrix(&tc);
		mm_set_array(&tc);
		mm_set_real(&tc);
		mm_set_general(&tc);

		if (mm_write_banner(f, tc) != 0)
			throw std::runtime_error("Error writing Matrix Market banner.");

		int M = mat.rows();
		int N = mat.cols();
		if (mm_write_mtx_array_size(f, M, N) != 0)
			throw std::runtime_error("Error writing matrix size.");

		// Matrix Market stores array data column-wise (Fortran order)
		for (int j = 0; j < N; ++j) {
			for (int i = 0; i < M; ++i) {
				if (fprintf(f, "%.16g\n", *mat(i, j)) < 0)
					throw std::runtime_error("Error writing matrix value.");
			}
		}

		fclose(f);
	}
};

}  // namespace swnumeric
