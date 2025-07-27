#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

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

void test_static_tensor_allocation() {
	TestFramework::start_test("Static Tensor Allocation");

	Tensor<double, 2, 3> t;
	TestFramework::assert_true(t.isAlloced(), "tensor is allocated");
	TestFramework::assert_true(t.isStaticSized(), "tensor is statically sized");
	TestFramework::assert_true(t.size() == 6, "tensor size == 6");
	TestFramework::assert_true(t.comptimeSize() == 6, "comptimeSize == 6");
}

void test_tensor_setters() {
	TestFramework::start_test("Setters on Static Tensor");

	Tensor<double, 2, 3> t;
	t.setZero();
	for (size_t i = 0; i < t.size(); i++) {
		TestFramework::assert_equal(t[i], 0.0, "setZero sets to 0");
	}

	t.setOne();
	for (size_t i = 0; i < t.size(); i++) {
		TestFramework::assert_equal(t[i], 1.0, "setOne sets to 1");
	}

	t.setConstant(3.14);
	for (size_t i = 0; i < t.size(); i++) {
		TestFramework::assert_equal(t[i], 3.14, "setConstant sets to 3.14");
	}
}

void test_tensor_assignment_operator() {
	TestFramework::start_test("Assignment between same-shape tensors");

	Tensor<double, 2, 3> a;
	Tensor<double, 2, 3> b;
	b.setConstant(5.0);

	a = b;
	for (size_t i = 0; i < a.size(); i++) {
		TestFramework::assert_equal(a[i], 5.0, "assigned value matches");
	}
}

void test_shape_mismatch_exception() {
	TestFramework::start_test("Shape mismatch throws on assignment");

	Tensor<double, 2, 3> a;
	Tensor<double, 3, 2> b;

	TestFramework::assert_throws(
		[&]() {
			a = b;	// Should throw
		},
		"assignment with mismatched shape");
}

void test_dynamic_tensor() {
	TestFramework::start_test("Dynamic Tensor Allocation");

	std::array<uint32_t, 2> shape = {3, 4};
	Tensor<double, 0, 0> t(shape);

	TestFramework::assert_true(t.isAlloced(), "tensor is allocated");
	TestFramework::assert_true(!t.isStaticSized(), "tensor is dynamic");
	TestFramework::assert_true(t.size() == 12, "tensor size == 12");

	t.setOne();
	for (size_t i = 0; i < t.size(); i++) {
		TestFramework::assert_equal(t[i], 1.0, "setOne works on dynamic");
	}
}

void run_all_tensor_tests() {
	test_static_tensor_allocation();
	test_tensor_setters();
	test_tensor_assignment_operator();
	test_shape_mismatch_exception();
	test_dynamic_tensor();

	TestFramework::print_summary();
}

// Reuse test framework (assumed included already)

void test_inplace_scalar_ops() {
	TestFramework::start_test("In-place scalar operations");

	Tensor<double, 2, 2> t;
	t.setConstant(2.0);

	t += 3.0;
	for (size_t i = 0; i < t.size(); i++)
		TestFramework::assert_equal(t[i], 5.0, "+= scalar works");

	t -= 1.0;
	for (size_t i = 0; i < t.size(); i++)
		TestFramework::assert_equal(t[i], 4.0, "-= scalar works");

	t *= 2.0;
	for (size_t i = 0; i < t.size(); i++)
		TestFramework::assert_equal(t[i], 8.0, "*= scalar works");

	t /= 2.0;
	for (size_t i = 0; i < t.size(); i++)
		TestFramework::assert_equal(t[i], 4.0, "/= scalar works");
}

void test_inplace_tensor_ops() {
	TestFramework::start_test("In-place elementwise tensor ops");

	Tensor<double, 2, 2> a;
	Tensor<double, 2, 2> b;

	a.setConstant(5.0);
	b.setConstant(3.0);

	a += b;
	for (size_t i = 0; i < a.size(); i++)
		TestFramework::assert_equal(a[i], 8.0, "tensor += tensor");

	a -= b;
	for (size_t i = 0; i < a.size(); i++)
		TestFramework::assert_equal(a[i], 5.0, "tensor -= tensor");

	a *= b;
	for (size_t i = 0; i < a.size(); i++)
		TestFramework::assert_equal(a[i], 15.0, "tensor *= tensor");

	a /= b;
	for (size_t i = 0; i < a.size(); i++)
		TestFramework::assert_equal(a[i], 5.0, "tensor /= tensor");
}

void test_STOP_expression_scalar_add() {
	TestFramework::start_test("STOP scalar expression: Add");

	Tensor<double, 2, 2> a;
	a.setConstant(2.0);

	auto expr =
		swnumeric::impl::STOP<double, swnumeric::impl::OPType::Add, 2, 2>{a,
																		  3.0};

	Tensor<double, 2, 2> b;
	b = expr;

	for (size_t i = 0; i < b.size(); i++)
		TestFramework::assert_equal(b[i], 5.0, "b = a + 3 (via STOP)");
}

void test_STOP_inplace_add() {
	TestFramework::start_test("STOP in-place += expression");

	Tensor<double, 2, 2> a;
	a.setConstant(2.0);

	auto expr =
		swnumeric::impl::STOP<double, swnumeric::impl::OPType::Mul, 2, 2>{a,
																		  4.0};

	Tensor<double, 2, 2> b;
	b.setConstant(1.0);

	b += expr;

	for (size_t i = 0; i < b.size(); i++)
		TestFramework::assert_equal(b[i], 9.0, "b += 4*a (via STOP)");
}

/**
void test_double_alloc_prevention() {
	TestFramework::start_test("Double allocation prevention");

	Tensor<double, 2, 2> t;

	TestFramework::assert_throws(
		[&]() {
			std::array<uint32_t, 2> shape = {2, 2};
			t = Tensor<double, 0, 0>(shape);  // implicit alloc
		},
		"allocating into an already-allocated tensor");
}
*/

void test_free_and_reuse() {
	TestFramework::start_test("Free tensor and reallocate");

	Tensor<double, 0, 0> t;
	std::array<uint32_t, 2> s1 = {2, 2};
	t = Tensor<double, 0, 0>(s1);

	t.setConstant(2.0);
	t.free();

	TestFramework::assert_true(!t.isAlloced(), "tensor is no longer allocated");

	std::array<uint32_t, 2> s2 = {3, 1};
	t = Tensor<double, 0, 0>(s2);
	t.setOne();

	TestFramework::assert_true(t.size() == 3, "tensor resized to 3");
	for (size_t i = 0; i < t.size(); i++)
		TestFramework::assert_equal(t[i], 1.0,
									"value preserved after reallocation");
}

int main() {
	std::cout << "=== swnumeric::Tensor Test Suite ===" << std::endl;
	std::cout << "Testing Tensor class implementation..." << std::endl
			  << std::endl;

	try {
		test_static_tensor_allocation();
		test_tensor_setters();
		test_tensor_assignment_operator();
		test_shape_mismatch_exception();
		test_dynamic_tensor();

		test_inplace_scalar_ops();
		test_inplace_tensor_ops();
		test_STOP_expression_scalar_add();
		test_STOP_inplace_add();
		//	test_double_alloc_prevention();
		test_free_and_reuse();

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
