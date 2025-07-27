#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include "../Tensor.hpp"  // Your Matrix/Vector header

// === Reuse TestFramework ===
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

// === Test Functions ===

void test_matrix_basic_construction() {
	TestFramework::start_test("Matrix Basic Construction");

	swnumeric::Matrix22 m1;
	TestFramework::assert_equal(m1.rows(), 2, "Matrix22 rows");
	TestFramework::assert_equal(m1.cols(), 2, "Matrix22 cols");

	swnumeric::Matrix<0, 0> m2(3, 4);
	TestFramework::assert_equal(m2.rows(), 3, "Dynamic matrix rows");
	TestFramework::assert_equal(m2.cols(), 4, "Dynamic matrix cols");

	for (size_t i = 0; i < m1.rows(); ++i)
		for (size_t j = 0; j < m1.cols(); ++j)
			TestFramework::assert_equal(*m1(i, j), 0.0, "Default init = 0");
}

void test_matrix_element_access() {
	TestFramework::start_test("Matrix Element Access");

	swnumeric::Matrix33 m;
	*m(0, 0) = 1.0;
	*m(1, 1) = 2.0;
	*m(2, 2) = 3.0;

	TestFramework::assert_equal(*m(0, 0), 1.0, "Access (0,0)");
	TestFramework::assert_equal(*m(1, 1), 2.0, "Access (1,1)");
	TestFramework::assert_equal(*m(2, 2), 3.0, "Access (2,2)");
}

void test_matrix_setters() {
	TestFramework::start_test("Matrix Setter Methods");

	swnumeric::Matrix33 m;
	m.setZero();
	for (size_t i = 0; i < m.rows(); ++i)
		for (size_t j = 0; j < m.cols(); ++j)
			TestFramework::assert_equal(*m(i, j), 0.0, "setZero");

	m.setOne();
	for (size_t i = 0; i < m.rows(); ++i)
		for (size_t j = 0; j < m.cols(); ++j)
			TestFramework::assert_equal(*m(i, j), 1.0, "setOne");

	m.setConstant(3.14);
	for (size_t i = 0; i < m.rows(); ++i)
		for (size_t j = 0; j < m.cols(); ++j)
			TestFramework::assert_equal(*m(i, j), 3.14, "setConstant");
}

void test_matrix_alias_types() {
	TestFramework::start_test("Matrix Alias Types");

	swnumeric::Matrix22 m2;
	swnumeric::Matrix33 m3;
	swnumeric::Matrix44 m4;

	TestFramework::assert_equal(m2.rows(), 2, "Matrix22 size");
	TestFramework::assert_equal(m3.cols(), 3, "Matrix33 size");
	TestFramework::assert_equal(m4.rows(), 4, "Matrix44 size");

	*m2(0, 0) = 5.0;
	*m2(1, 1) = 7.0;
	TestFramework::assert_equal(*m2(0, 0), 5.0, "Matrix22 value [0,0]");
	TestFramework::assert_equal(*m2(1, 1), 7.0, "Matrix22 value [1,1]");
}

void test_matrix_arithmetic_operations() {
	TestFramework::start_test("Matrix Arithmetic Operations");

	swnumeric::Matrix22 a, b;
	a.setConstant(2.0);
	b.setConstant(3.0);

	swnumeric::Matrix22 c;
	c = a + b;
	TestFramework::assert_equal(*c(0, 0), 5.0, "Matrix addition");
	TestFramework::assert_equal(*c(1, 1), 5.0, "Matrix addition");

	c = a - b;
	TestFramework::assert_equal(*c(0, 0), -1.0, "Matrix subtraction");

	c = a * 2.0;
	TestFramework::assert_equal(*c(0, 0), 4.0, "Scalar multiplication");

	c = b / 3.0;
	TestFramework::assert_equal(*c(0, 0), 1.0, "Scalar division");
}

void test_dynamic_matrix_operations() {
	TestFramework::start_test("Dynamic Matrix Operations");

	swnumeric::Matrix<0, 0> m1(2, 2);
	*m1(0, 0) = 1.0;
	*m1(0, 1) = 2.0;
	*m1(1, 0) = 3.0;
	*m1(1, 1) = 4.0;

	auto m2 = m1;
	m2 *= 2.0;
	TestFramework::assert_equal(*m2(0, 0), 2.0, "Scalar mul");
	TestFramework::assert_equal(*m2(1, 1), 8.0, "Scalar mul");

	m2 += m1;
	TestFramework::assert_equal(*m2(0, 0), 3.0, "Matrix add");
	TestFramework::assert_equal(*m2(1, 1), 12.0, "Matrix add");
}

void test_matrix_edge_cases() {
	TestFramework::start_test("Matrix Edge Cases");

	swnumeric::Matrix<0, 0> empty;
	TestFramework::assert_equal(empty.rows(), 0, "Empty matrix rows");
	TestFramework::assert_equal(empty.cols(), 0, "Empty matrix cols");

	swnumeric::Matrix<0, 0> tiny(1, 1);
	*tiny(0, 0) = 42.0;
	TestFramework::assert_equal(*tiny(0, 0), 42.0, "1x1 matrix value");
}

// === Main Entry ===
int main() {
	std::cout << "=== swnumeric::Matrix Test Suite ===\n" << std::endl;

	try {
		test_matrix_basic_construction();
		test_matrix_element_access();
		test_matrix_setters();
		test_matrix_alias_types();
		test_matrix_arithmetic_operations();
		test_dynamic_matrix_operations();
		test_matrix_edge_cases();

		TestFramework::print_summary();

		return (TestFramework::tests_run == TestFramework::tests_passed) ? 0
																		 : 1;
	} catch (const std::exception& e) {
		std::cerr << "Test suite failed with exception: " << e.what()
				  << std::endl;
		return 1;
	} catch (...) {
		std::cerr << "Test suite";
	}
}
