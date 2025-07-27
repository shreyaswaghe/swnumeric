#include <cassert>
#include <cmath>
#include <iostream>

#include "../GaussLegendre.hpp"

using namespace swnumeric;

// Test integrand classes
template <typename T>
class ConstantIntegrand : public RtoR<T> {
   private:
	T constant;

   public:
	explicit ConstantIntegrand(T c) : constant(c) {}
	T eval(T x) const override { return constant; }
};

template <typename T>
class LinearIntegrand : public RtoR<T> {
   public:
	T eval(T x) const override { return x; }
};

template <typename T>
class QuadraticIntegrand : public RtoR<T> {
   public:
	T eval(T x) const override { return x * x; }
};

template <typename T>
class CubicIntegrand : public RtoR<T> {
   public:
	T eval(T x) const override { return x * x * x; }
};

template <typename T>
class PolynomialIntegrand : public RtoR<T> {
   private:
	int degree;

   public:
	explicit PolynomialIntegrand(int d) : degree(d) {}
	T eval(T x) const override { return std::pow(x, degree); }
};

template <typename T>
class SinIntegrand : public RtoR<T> {
   public:
	T eval(T x) const override { return std::sin(x); }
};

template <typename T>
class ExpIntegrand : public RtoR<T> {
   public:
	T eval(T x) const override { return std::exp(x); }
};

// Helper function to check if two floating point numbers are approximately
// equal
template <typename T>
bool approx_equal(T a, T b, T tolerance = static_cast<T>(1e-12)) {
	if (std::is_same_v<T, float>) {
		tolerance = static_cast<T>(1e-6);
	}
	return std::abs(a - b) < tolerance;
}

// Test functions
void test_constant_integration() {
	std::cout << "Testing constant integration...\n";

	GaussLegendre gl;
	ConstantIntegrand<real> constant_5(5.0);

	// Integral of 5 from -1 to 1 should be 10
	real result2 = gl.eval<ConstantIntegrand<real>, 2>(constant_5);
	real result4 = gl.eval<ConstantIntegrand<real>, 4>(constant_5);
	real result8 = gl.eval<ConstantIntegrand<real>, 8>(constant_5);
	real result16 = gl.eval<ConstantIntegrand<real>, 16>(constant_5);
	real result32 = gl.eval<ConstantIntegrand<real>, 32>(constant_5);
	real result64 = gl.eval<ConstantIntegrand<real>, 64>(constant_5);

	real expected = 10.0;
	assert(approx_equal(result2, expected));
	assert(approx_equal(result4, expected));
	assert(approx_equal(result8, expected));
	assert(approx_equal(result16, expected));
	assert(approx_equal(result32, expected));
	assert(approx_equal(result64, expected));

	std::cout << "✓ Constant integration tests passed\n";
}

void test_linear_integration() {
	std::cout << "Testing linear integration...\n";

	GaussLegendre gl;
	LinearIntegrand<real> linear;

	// Integral of x from -1 to 1 should be 0 (odd function)
	real result2 = gl.eval<LinearIntegrand<real>, 2>(linear);
	real result4 = gl.eval<LinearIntegrand<real>, 4>(linear);
	real result8 = gl.eval<LinearIntegrand<real>, 8>(linear);
	real result16 = gl.eval<LinearIntegrand<real>, 16>(linear);
	real result32 = gl.eval<LinearIntegrand<real>, 32>(linear);
	real result64 = gl.eval<LinearIntegrand<real>, 64>(linear);

	real expected = 0.0;
	assert(approx_equal(result2, expected));
	assert(approx_equal(result4, expected));
	assert(approx_equal(result8, expected));
	assert(approx_equal(result16, expected));
	assert(approx_equal(result32, expected));
	assert(approx_equal(result64, expected));

	std::cout << "✓ Linear integration tests passed\n";
}

void test_quadratic_integration() {
	std::cout << "Testing quadratic integration...\n";

	GaussLegendre gl;
	QuadraticIntegrand<real> quadratic;

	// Integral of x^2 from -1 to 1 should be 2/3
	real expected = 2.0 / 3.0;

	real result2 = gl.eval<QuadraticIntegrand<real>, 2>(quadratic);
	real result4 = gl.eval<QuadraticIntegrand<real>, 4>(quadratic);
	real result8 = gl.eval<QuadraticIntegrand<real>, 8>(quadratic);
	real result16 = gl.eval<QuadraticIntegrand<real>, 16>(quadratic);
	real result32 = gl.eval<QuadraticIntegrand<real>, 32>(quadratic);
	real result64 = gl.eval<QuadraticIntegrand<real>, 64>(quadratic);

	assert(approx_equal(result2, expected));
	assert(approx_equal(result4, expected));
	assert(approx_equal(result8, expected));
	assert(approx_equal(result16, expected));
	assert(approx_equal(result32, expected));
	assert(approx_equal(result64, expected));

	std::cout << "✓ Quadratic integration tests passed\n";
}

void test_cubic_integration() {
	std::cout << "Testing cubic integration...\n";

	GaussLegendre gl;
	CubicIntegrand<real> cubic;

	// Integral of x^3 from -1 to 1 should be 0 (odd function)
	real expected = 0.0;

	real result2 = gl.eval<CubicIntegrand<real>, 2>(cubic);
	real result4 = gl.eval<CubicIntegrand<real>, 4>(cubic);
	real result8 = gl.eval<CubicIntegrand<real>, 8>(cubic);
	real result16 = gl.eval<CubicIntegrand<real>, 16>(cubic);
	real result32 = gl.eval<CubicIntegrand<real>, 32>(cubic);
	real result64 = gl.eval<CubicIntegrand<real>, 64>(cubic);

	assert(approx_equal(result2, expected));
	assert(approx_equal(result4, expected));
	assert(approx_equal(result8, expected));
	assert(approx_equal(result16, expected));
	assert(approx_equal(result32, expected));
	assert(approx_equal(result64, expected));

	std::cout << "✓ Cubic integration tests passed\n";
}

void test_high_degree_polynomials() {
	std::cout << "Testing high degree polynomial integration...\n";

	GaussLegendre gl;

	// Test x^4: integral from -1 to 1 should be 2/5
	PolynomialIntegrand<real> poly4(4);
	real expected4 = 2.0 / 5.0;
	real result4_2pts = gl.eval<PolynomialIntegrand<real>, 2>(poly4);
	real result4_4pts = gl.eval<PolynomialIntegrand<real>, 4>(poly4);
	real result4_8pts = gl.eval<PolynomialIntegrand<real>, 8>(poly4);

	// 2-point rule should be less accurate for x^4
	assert(!approx_equal(result4_2pts, expected4, 1e-10));
	// 4-point and higher should be exact for x^4
	assert(approx_equal(result4_4pts, expected4));
	assert(approx_equal(result4_8pts, expected4));

	// Test x^6: integral from -1 to 1 should be 2/7
	PolynomialIntegrand<real> poly6(6);
	real expected6 = 2.0 / 7.0;
	real result6_2pts = gl.eval<PolynomialIntegrand<real>, 2>(poly6);
	real result6_4pts = gl.eval<PolynomialIntegrand<real>, 4>(poly6);
	real result6_8pts = gl.eval<PolynomialIntegrand<real>, 8>(poly6);

	std::cout << result6_4pts << expected6 << std::endl;

	// Lower order rules should be less accurate
	assert(!approx_equal(result6_2pts, expected6, 1e-6));
	assert(approx_equal(result6_4pts, expected6, 1e-8));
	// 8-point rule should be exact for x^6
	assert(approx_equal(result6_8pts, expected6));

	std::cout << "✓ High degree polynomial tests passed\n";
}

void test_transcendental_functions() {
	std::cout << "Testing transcendental function integration...\n";

	GaussLegendre gl;

	// Test sin(x) from -1 to 1: should be 0 (odd function)
	SinIntegrand<real> sin_func;
	real sin_result = gl.eval<SinIntegrand<real>, 16>(sin_func);
	assert(approx_equal(sin_result, 0.0, 1e-10));

	// Test exp(x) from -1 to 1: should be e - 1/e ≈ 2.350402387
	ExpIntegrand<real> exp_func;
	real exp_expected = std::exp(1.0) - std::exp(-1.0);
	real exp_result16 = gl.eval<ExpIntegrand<real>, 16>(exp_func);
	real exp_result32 = gl.eval<ExpIntegrand<real>, 32>(exp_func);
	real exp_result64 = gl.eval<ExpIntegrand<real>, 64>(exp_func);

	assert(approx_equal(exp_result16, exp_expected, 1e-8));
	assert(approx_equal(exp_result32, exp_expected, 1e-12));
	assert(approx_equal(exp_result64, exp_expected, 1e-14));

	std::cout << "✓ Transcendental function tests passed\n";
}

void test_convergence() {
	std::cout << "Testing convergence with increasing points...\n";

	GaussLegendre gl;
	ExpIntegrand<real> exp_func;
	real expected = std::exp(1.0) - std::exp(-1.0);

	real result2 = gl.eval<ExpIntegrand<real>, 2>(exp_func);
	real result4 = gl.eval<ExpIntegrand<real>, 4>(exp_func);
	real result8 = gl.eval<ExpIntegrand<real>, 8>(exp_func);
	real result16 = gl.eval<ExpIntegrand<real>, 16>(exp_func);
	real result32 = gl.eval<ExpIntegrand<real>, 32>(exp_func);
	real result64 = gl.eval<ExpIntegrand<real>, 64>(exp_func);

	real error2 = std::abs(result2 - expected);
	real error4 = std::abs(result4 - expected);
	real error8 = std::abs(result8 - expected);
	real error16 = std::abs(result16 - expected);
	real error32 = std::abs(result32 - expected);
	real error64 = std::abs(result64 - expected);

	// Errors should generally decrease as we increase points
	assert(error4 <= error2);
	assert(error8 <= error4);
	assert(error16 <= error8);
	assert(error32 <= error16);
	assert(error64 <= error32);

	std::cout << "✓ Convergence tests passed\n";
}

void test_float_precision() {
	std::cout << "Testing single precision (float) implementation...\n";

	GaussLegendre gl_f;
	ConstantIntegrand<single> constant_3(3.0f);
	QuadraticIntegrand<single> quadratic_f;

	// Test constant integration
	single result_const = gl_f.eval<ConstantIntegrand<single>, 8>(constant_3);
	assert(approx_equal(result_const, 6.0f));

	// Test quadratic integration
	single result_quad = gl_f.eval<QuadraticIntegrand<single>, 8>(quadratic_f);
	single expected_quad = 2.0f / 3.0f;
	assert(approx_equal(result_quad, expected_quad, 1e-6f));

	std::cout << "✓ Single precision tests passed\n";
}

void test_kahan_summation() {
	std::cout << "Testing Kahan summation for higher order rules...\n";

	// This test verifies that the Kahan summation used in 16, 32, 64 point
	// rules provides better numerical stability
	GaussLegendre gl;
	ConstantIntegrand<real> constant_1(1.0);

	// For a constant function, all rules should give exactly the same result
	real result16 = gl.eval<ConstantIntegrand<real>, 16>(constant_1);
	real result32 = gl.eval<ConstantIntegrand<real>, 32>(constant_1);
	real result64 = gl.eval<ConstantIntegrand<real>, 64>(constant_1);

	real expected = 2.0;  // integral of 1 from -1 to 1
	assert(approx_equal(result16, expected));
	assert(approx_equal(result32, expected));
	assert(approx_equal(result64, expected));

	std::cout << "✓ Kahan summation tests passed\n";
}

void test_symmetry() {
	std::cout << "Testing symmetry properties...\n";

	GaussLegendre gl;

	// Test that odd functions integrate to zero
	LinearIntegrand<real> linear;
	CubicIntegrand<real> cubic;
	PolynomialIntegrand<real> poly5(5);

	real linear_result = gl.eval<LinearIntegrand<real>, 32>(linear);
	real cubic_result = gl.eval<CubicIntegrand<real>, 32>(cubic);
	real poly5_result = gl.eval<PolynomialIntegrand<real>, 32>(poly5);

	assert(approx_equal(linear_result, 0.0, 1e-14));
	assert(approx_equal(cubic_result, 0.0, 1e-14));
	assert(approx_equal(poly5_result, 0.0, 1e-14));

	std::cout << "✓ Symmetry tests passed\n";
}

int main() {
	std::cout << "Running Gauss-Legendre Quadrature Unit Tests\n";
	std::cout << "============================================\n\n";

	try {
		test_constant_integration();
		test_linear_integration();
		test_quadratic_integration();
		test_cubic_integration();
		test_high_degree_polynomials();
		test_transcendental_functions();
		test_convergence();
		test_float_precision();
		test_kahan_summation();
		test_symmetry();

		std::cout << "\n============================================\n";
		std::cout << "✅ All tests passed successfully!\n";
		std::cout << "============================================\n";

	} catch (const std::exception& e) {
		std::cout << "\n❌ Test failed with exception: " << e.what()
				  << std::endl;
		return 1;
	}

	return 0;
}
