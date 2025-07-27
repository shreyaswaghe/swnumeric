#include <functional>
#include <iostream>

#include "../Tensor.hpp"  // Assuming your code is in vector.h

// Simple test framework
class TestFramework {
   public:
	static int tests_run;
	static int tests_passed;
	static std::string current_test;

	static void start_test(const std::string& name) {
		current_test = name;
		std::cout << "Running test: " << name << std::endl;
	}

	static void assert_true(bool condition, const std::string& message = "") {
		tests_run++;
		if (condition) {
			tests_passed++;
			std::cout << "  ✓ PASS";
			if (!message.empty()) std::cout << ": " << message;
			std::cout << std::endl;
		} else {
			std::cout << "  ✗ FAIL";
			if (!message.empty()) std::cout << ": " << message;
			std::cout << std::endl;
		}
	}

	static void assert_equal(double a, double b,
							 const std::string& message = "",
							 double epsilon = 1e-10) {
		assert_true(std::abs(a - b) < epsilon,
					message + " (expected: " + std::to_string(a) +
						", got: " + std::to_string(b) + ")");
	}

	static void assert_throws(std::function<void()> func,
							  const std::string& message = "") {
		tests_run++;
		try {
			func();
			std::cout << "  ✗ FAIL: " << message
					  << " (expected exception but none thrown)" << std::endl;
		} catch (...) {
			tests_passed++;
			std::cout << "  ✓ PASS: " << message
					  << " (exception thrown as expected)" << std::endl;
		}
	}

	static void print_summary() {
		std::cout << "\n=== Test Summary ===" << std::endl;
		std::cout << "Tests run: " << tests_run << std::endl;
		std::cout << "Tests passed: " << tests_passed << std::endl;
		std::cout << "Tests failed: " << (tests_run - tests_passed)
				  << std::endl;
		std::cout << "Success rate: "
				  << (tests_run > 0 ? (100.0 * tests_passed / tests_run) : 0)
				  << "%" << std::endl;
	}
};

int TestFramework::tests_run = 0;
int TestFramework::tests_passed = 0;
std::string TestFramework::current_test = "";

// Test helper functions
template <size_t N>
void print_vector(const swnumeric::Vector<N>& v, const std::string& name) {
	std::cout << "  " << name << ": [";
	for (size_t i = 0; i < v.size(); i++) {
		std::cout << v[i];
		if (i < v.size() - 1) std::cout << ", ";
	}
	std::cout << "]" << std::endl;
}

bool vectors_equal(const swnumeric::Vector<0>& a, const swnumeric::Vector<0>& b,
				   double epsilon = 1e-10) {
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); i++) {
		if (std::abs(a[i] - b[i]) > epsilon) return false;
	}
	return true;
}

template <size_t N>
bool vectors_equal(const swnumeric::Vector<N>& a, const swnumeric::Vector<N>& b,
				   double epsilon = 1e-10) {
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); i++) {
		if (std::abs(a[i] - b[i]) > epsilon) return false;
	}
	return true;
}

using namespace swnumeric;

void test_vector_indexing() {
	TestFramework::start_test("Vector indexing and element access");

	Vector<5, double> v;
	v.setConstant(3.14);

	for (size_t i = 0; i < 5; ++i) {
		TestFramework::assert_equal(v[i], 3.14, "operator[] access correct");
		TestFramework::assert_equal(*v(i), 3.14, "operator() access correct");
	}
}

void test_vector_constructor_and_size() {
	TestFramework::start_test("Vector constructor with size param");

	Vector<0, double> v(7);
	TestFramework::assert_true(v.size() == 7, "dynamic vector constructed");

	v.setOne();
	for (size_t i = 0; i < v.size(); i++) {
		TestFramework::assert_equal(v[i], 1.0, "setOne works");
	}
}

void test_vector_static_vs_dynamic_allocation() {
	TestFramework::start_test("Static vs dynamic Vector allocation");

	Vector<5, double> v_static;
	TestFramework::assert_true(v_static.isAlloced(),
							   "static vector is allocated");
	TestFramework::assert_true(v_static.isStaticSized(),
							   "static vector is statically sized");
	TestFramework::assert_true(v_static.size() == 5,
							   "static vector has correct size");

	Vector<0, double> v_dyn(5);
	TestFramework::assert_true(v_dyn.isAlloced(),
							   "dynamic vector is allocated");
	TestFramework::assert_true(!v_dyn.isStaticSized(),
							   "dynamic vector is dynamic sized");
	TestFramework::assert_true(v_dyn.size() == 5,
							   "dynamic vector has correct size");
}

void test_vector_setters() {
	TestFramework::start_test("Vector value setters");

	Vector<4, double> v;
	v.setZero();
	for (size_t i = 0; i < v.size(); i++)
		TestFramework::assert_equal(v[i], 0.0, "setZero works");

	v.setOne();
	for (size_t i = 0; i < v.size(); i++)
		TestFramework::assert_equal(v[i], 1.0, "setOne works");

	v.setConstant(3.1415);
	for (size_t i = 0; i < v.size(); i++)
		TestFramework::assert_equal(v[i], 3.1415, "setConstant works");
}

void test_vector_assignment_aliasing() {
	TestFramework::start_test("Vector assignment and aliasing");

	Vector<3, double> a;
	a.setConstant(2.0);

	Vector<3, double> b;
	b = a;

	a[0] = 99.0;
	TestFramework::assert_equal(b[0], 2.0, "assignment does not alias memory");
}

void test_vector_stop_scalar_ops() {
	TestFramework::start_test("Vector scalar expression (STOP)");

	Vector<4, double> v;
	v.setConstant(2.0);

	using namespace swnumeric::impl;
	auto expr = STOP<double, OPType::Mul, 4>{v, 3.0};

	Vector<4, double> out;
	out = expr;

	for (size_t i = 0; i < out.size(); i++)
		TestFramework::assert_equal(out[i], 6.0, "scalar multiplication works");
}

void test_vector_as_row_matrix() {
	TestFramework::start_test("Vector::asRowMatrix");

	Vector<4, double> v;
	for (size_t i = 0; i < 4; i++) v[i] = static_cast<double>(i + 1);

	auto m = v.asRowMatrix();  // Matrix<1, 4>
	for (size_t i = 0; i < 4; i++) {
		TestFramework::assert_equal(m[i], v[i], "asRowMatrix element matches");
	}
}

void test_vector_as_diagonal_matrix() {
	TestFramework::start_test("Vector::asDiagonalMatrix");

	Vector<3, double> v;
	v[0] = 2.0;
	v[1] = 4.0;
	v[2] = 6.0;

	auto m = v.asDiagonalMatrix();	// Matrix<3, 3>
	for (size_t i = 0; i < 3; i++) {
		for (size_t j = 0; j < 3; j++) {
			double expected = (i == j) ? v[i] : 0.0;
			TestFramework::assert_equal(m[i * 3 + j], expected,
										"asDiagonalMatrix[" +
											std::to_string(i) + "," +
											std::to_string(j) + "] correct");
		}
	}
}

void test_vector_matrix_round_trip() {
	TestFramework::start_test("Vector row-diagonal matrix round trip");

	Vector<3, double> v;
	v[0] = 1.0;
	v[1] = 2.0;
	v[2] = 3.0;

	auto diag = v.asDiagonalMatrix();
	auto row = v.asRowMatrix();

	TestFramework::assert_equal(diag[0], 1.0, "diag[0,0] correct");
	TestFramework::assert_equal(diag[4], 2.0, "diag[1,1] correct");
	TestFramework::assert_equal(diag[8], 3.0, "diag[2,2] correct");

	for (size_t i = 0; i < 3; i++)
		TestFramework::assert_equal(row[i], v[i], "row matrix copy correct");
}

void test_matrix_construction_and_shape() {
	TestFramework::start_test("Matrix construction and shape");

	Matrix<2, 3, double> m(2, 3);
	TestFramework::assert_true(m.rows() == 2, "Matrix rows == 2");
	TestFramework::assert_true(m.cols() == 3, "Matrix cols == 3");
	TestFramework::assert_true(m.size() == 6, "Matrix size == 6");
}

void test_matrix_indexing() {
	TestFramework::start_test("Matrix indexing (col-major)");

	Matrix<2, 2, double> m(2, 2);
	m[0] = 1.0;
	m[1] = 2.0;
	m[2] = 3.0;
	m[3] = 4.0;

	TestFramework::assert_equal(*m(0, 0), 1.0, "m(0,0) == 1");
	TestFramework::assert_equal(*m(1, 0), 2.0, "m(1,0) == 2");
	TestFramework::assert_equal(*m(0, 1), 3.0, "m(0,1) == 3");
	TestFramework::assert_equal(*m(1, 1), 4.0, "m(1,1) == 4");
}

void test_matrix_assignment_stop_expr() {
	TestFramework::start_test("Matrix assignment with STOP expression");

	Matrix<2, 2, double> m(2, 2);
	m.setConstant(2.0);

	impl::STOP<double, impl::OPType::Mul, 2, 2> expr = {m, 3.0};
	Matrix<2, 2, double> out(2, 2);
	out = expr;

	for (size_t i = 0; i < out.size(); i++) {
		TestFramework::assert_equal(out[i], 6.0, "out[i] == 6.0");
	}
}

void test_matrix_diagonalAsVector() {
	TestFramework::start_test("Matrix::diagonalAsVector");

	Matrix<3, 3, double> m(3, 3);
	for (size_t i = 0; i < 9; i++) m[i] = static_cast<double>(i + 1);

	auto diag = m.diagonalAsVector();
	TestFramework::assert_equal(diag[0], 1.0, "diag[0] == 1");
	TestFramework::assert_equal(diag[1], 5.0, "diag[1] == 5");
	TestFramework::assert_equal(diag[2], 9.0, "diag[2] == 9");
}

void test_matrix_row_col_accessors() {
	TestFramework::start_test("Matrix::row() and Matrix::col()");

	Matrix<3, 3, double> m(3, 3);
	double val = 1.0;
	for (size_t i = 0; i < 9; i++) m[i] = val++;

	Vector<3, double> row1 = m.row(1);	// Second row (index 1)
	TestFramework::assert_equal(row1[0], m[1], "row1[0] == m[1]");
	TestFramework::assert_equal(row1[1], m[4], "row1[1] == m[4]");
	TestFramework::assert_equal(row1[2], m[7], "row1[2] == m[7]");

	Vector<3, double> col0 = m.col(2);	// Third row (index 2)
	TestFramework::assert_equal(col0[0], m[6], "col0[0] == m[3]");
	TestFramework::assert_equal(col0[1], m[7], "col0[1] == m[4]");
	TestFramework::assert_equal(col0[2], m[8], "col0[2] == m[5]");
}

int main() {
	try {
		std::cout << "\n=== swnumeric::Vector Test Suite ===" << std::endl;

		test_vector_indexing();
		test_vector_constructor_and_size();
		test_vector_static_vs_dynamic_allocation();
		test_vector_setters();
		test_vector_assignment_aliasing();
		test_vector_stop_scalar_ops();
		test_vector_as_row_matrix();
		test_vector_as_diagonal_matrix();
		test_vector_matrix_round_trip();

		test_matrix_construction_and_shape();
		test_matrix_indexing();
		test_matrix_assignment_stop_expr();
		test_matrix_diagonalAsVector();
		test_matrix_row_col_accessors();

		TestFramework::print_summary();

		return (TestFramework::tests_run == TestFramework::tests_passed) ? 0
																		 : 1;
	} catch (const std::exception& e) {
		std::cerr << "Test suite failed with exception: " << e.what()
				  << std::endl;
		return 1;
	} catch (...) {
		std::cerr << "Test suite failed with unknown exception" << std::endl;
		return 1;
	}
}
