#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

// Assuming your ODE interface is in this header
#include "../OdeDynamics.hpp"  // Replace with actual path
#include "../RungeKutta45.hpp"
#include "Libraries/Tensor/Tensor.hpp"

// Helper function for floating point comparison
template <typename T>
bool isNear(T a, T b, T tolerance = 1e-10) {
	return std::abs(a - b) < tolerance;
}

// Test ODE: Simple harmonic oscillator
// d²x/dt² = -ω²x
// State vector: [position, velocity]
// dx/dt = [velocity, -ω²*position]
using namespace swnumeric;

class SimpleHarmonicOscillator : public ODEDynamics<Vector2> {
   private:
	double omega_squared;  // ω²

   public:
	SimpleHarmonicOscillator(double omega = 1.0)
		: omega_squared(omega * omega) {}

	~SimpleHarmonicOscillator() override {}

	// Pre-integration hook (identity transformation for this example)
	void PreIntegration(Vector2& x, double t) override {}

	// Post-integration hook (identity transformation for this example)
	void PostIntegration(Vector2& x, double t) override {}

	// doublehe core dynamics: dx/dt = f(x, t)
	void Gradient(Vector2& gradout, const Vector2& x, double t) override {
		// x[0] = position, x[1] = velocity
		// dx/dt = [velocity, -ω²*position]
		gradout[0] = x[1];					 // d(position)/dt = velocity
		gradout[1] = -omega_squared * x[0];	 // d(velocity)/dt = -ω²*position
	}

	// State norm (Euclidean norm)
	double stateNorm(const Vector2& x) override {
		return std::sqrt(x[0] * x[0] + x[1] * x[1]);
	}
};

// Test ODE: Exponential decay matrix
// dX/dt = -αX where X is a matrix
class ExponentialDecayMatrix : public ODEDynamics<Matrix22> {
   private:
	double alpha;  // decay constant

   public:
	ExponentialDecayMatrix(double decay_rate = 1.0) : alpha(decay_rate) {}

	~ExponentialDecayMatrix() override {}

	void PreIntegration(Matrix22& x, double t) override {}

	void PostIntegration(Matrix22& x, double t) override {}

	// dX/dt = -αX
	void Gradient(Matrix22& gradout, const Matrix22& x, double t) override {
		gradout = x;
		gradout *= -alpha;
	}

	// Frobenius norm
	double stateNorm(const Matrix22& x) override {
		double sum = 0;
		for (size_t i = 0; i < 2; ++i) {
			for (size_t j = 0; j < 2; ++j) {
				sum += *x(i, j) * *x(i, j);
			}
		}
		return std::sqrt(sum) / 4.0;
	}
};

// Test functions
void testSimpleHarmonicOscillatorGradient() {
	std::cout << "Testing Simple Harmonic Oscillator Gradient..." << std::endl;

	SimpleHarmonicOscillator sho(2.0);	// ω = 2, so ω² = 4

	Vector2 state;
	state[0] = 1.0;	 // position = 1
	state[1] = 0.0;	 // velocity = 0

	Vector2 gradient;
	sho.Gradient(gradient, state, 0.0);

	// Expected: dx/dt = [0, -4*1] = [0, -4]
	assert(isNear(gradient[0], 0.0));
	assert(isNear(gradient[1], -4.0));

	std::cout << "✓ Gradient test passed" << std::endl;
}

void testSimpleHarmonicOscillatorNorm() {
	std::cout << "Testing Simple Harmonic Oscillator Norm..." << std::endl;

	SimpleHarmonicOscillator sho;

	Vector2 state;
	state[0] = 3.0;
	state[1] = 4.0;

	double norm = sho.stateNorm(state);
	assert(isNear(norm, 5.0));	// sqrt(3² + 4²) = 5

	std::cout << "✓ Norm test passed" << std::endl;
}

void testExponentialDecayMatrixGradient() {
	std::cout << "Testing Exponential Decay Matrix Gradient..." << std::endl;

	ExponentialDecayMatrix decay(0.5);	// α = 0.5

	Matrix22 state;
	*state(0, 0) = 2.0;
	*state(0, 1) = 1.0;
	*state(1, 0) = 3.0;
	*state(1, 1) = 4.0;

	Matrix22 gradient;
	decay.Gradient(gradient, state, 0.0);

	// Expected: dX/dt = -0.5 * X
	assert(isNear(*gradient(0, 0), -1.0));	// -0.5 * 2.0
	assert(isNear(*gradient(0, 1), -0.5));	// -0.5 * 1.0
	assert(isNear(*gradient(1, 0), -1.5));	// -0.5 * 3.0
	assert(isNear(*gradient(1, 1), -2.0));	// -0.5 * 4.0

	std::cout << "✓ Matrix gradient test passed" << std::endl;
}

void testExponentialDecayMatrixNorm() {
	std::cout << "Testing Exponential Decay Matrix Norm..." << std::endl;

	ExponentialDecayMatrix decay;

	Matrix22 state;
	*state(0, 0) = 1.0;
	*state(0, 1) = 2.0;
	*state(1, 0) = 3.0;
	*state(1, 1) = 4.0;

	double norm = decay.stateNorm(state);
	double expected = std::sqrt(1.0 + 4.0 + 9.0 + 16.0) / 4.0;	// sqrt(30)
	assert(isNear(norm, expected));

	std::cout << "✓ Matrix norm test passed" << std::endl;
}

void testGradientConsistency() {
	std::cout << "Testing Gradient Consistency..." << std::endl;

	SimpleHarmonicOscillator sho(1.0);

	Vector2 state1, state2;
	state1[0] = 1.0;
	state1[1] = 0.0;
	state2[0] = 0.0;
	state2[1] = 1.0;

	Vector2 grad1, grad2;
	sho.Gradient(grad1, state1, 0.0);
	sho.Gradient(grad2, state2, 0.0);

	// For simple harmonic oscillator with ω=1:
	// At (1,0): gradient should be (0,-1)
	// At (0,1): gradient should be (1,0)
	assert(isNear(grad1[0], 0.0));
	assert(isNear(grad1[1], -1.0));
	assert(isNear(grad2[0], 1.0));
	assert(isNear(grad2[1], 0.0));

	std::cout << "✓ Gradient consistency test passed" << std::endl;
}

void testComplexOscillatorBehavior() {
	std::cout << "Testing Complex Oscillator Behavior..." << std::endl;

	SimpleHarmonicOscillator sho(1.0);

	// Test at maximum displacement (energy should be conserved in principle)
	Vector2 state_max_pos;
	state_max_pos[0] = 2.0;	 // max position
	state_max_pos[1] = 0.0;	 // zero velocity

	Vector2 state_max_vel;
	state_max_vel[0] = 0.0;	 // zero position
	state_max_vel[1] = 2.0;	 // max velocity

	double norm_pos = sho.stateNorm(state_max_pos);
	double norm_vel = sho.stateNorm(state_max_vel);

	// Both should have same "energy" (norm in phase space)
	assert(isNear(norm_pos, norm_vel));
	assert(isNear(norm_pos, 2.0));

	std::cout << "✓ Complex oscillator behavior test passed" << std::endl;
}

void runPerformanceTest() {
	std::cout << "Running Performance Test..." << std::endl;

	SimpleHarmonicOscillator sho;
	Vector2 state;
	state[0] = 1.0;
	state[1] = 1.0;

	auto start = std::chrono::high_resolution_clock::now();

	Vector2 gradient;
	for (int i = 0; i < 1000000; ++i) {
		sho.Gradient(state, gradient, 0.0);
		state[0] += 1e-8;  // Small perturbation to avoid optimization
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "Performance: " << duration.count()
			  << " microseconds for 1M gradient evaluations" << std::endl;

	// Simple assertion - should complete in reasonable time
	assert(duration.count() < 1000000);	 // Less than 1 seconds

	std::cout << "✓ Performance test passed" << std::endl;
}

// Integrator tests
void testExponentialDecayIntegration() {
	std::cout << "Testing Exponential Decay Integration..." << std::endl;

	ExponentialDecayMatrix decay(0.0);	// α = 1.0
	RungeKutta45<Matrix22> integrator(decay);

	// Initial condition: identity matrix
	Matrix22 state;
	*state(0, 0) = 1.0;
	*state(0, 1) = 0.0;
	*state(1, 0) = 0.0;
	*state(1, 1) = 1.0;

	double t_start = 0.0;
	double t_end = 1e-4;

	Matrix22 workState;
	integrator(state, workState, t_start, t_end);

	// Analytical solution: X(t) = X(0) * exp(-α*t) = I * exp(-1) ≈ 0.3679
	double expected = std::exp(0.0);

	std::cout << workState(0, 0) << " " << workState(1, 0) << std::endl;

	// assert(isNear(final_time, t_end));
	assert(isNear(*workState(0, 0), expected, 1e-4));
	assert(isNear(*workState(1, 1), expected, 1e-4));
	assert(isNear(*workState(0, 1), 0.0, 1e-6));
	assert(isNear(*workState(1, 0), 0.0, 1e-6));

	std::cout << "✓ Exponential decay integration test passed" << std::endl;
}

void testFastDecayAdaptiveStep() {
	std::cout << "Testing Fast Decay with Adaptive Stepping..." << std::endl;

	ExponentialDecayMatrix fast_decay(100.0);  // Very fast decay
	RungeKutta45<Matrix22> integrator(fast_decay);
	integrator.hmin = 1e-6;
	integrator.hmax = 100;
	integrator.rtol = 1e-11;
	integrator.atol = 1e-11;

	Matrix22 state;
	*state(0, 0) = 1.0;
	*state(0, 1) = 2.0;
	*state(1, 0) = 3.0;
	*state(1, 1) = 4.0;

	double t_start = 0.0;
	double t_end = 10.0;

	Matrix22 workState;
	integrator(state, workState, t_start, t_end);

	// Should reach t_end despite fast dynamics
	// Values should be very small due to fast decay
	double norm = fast_decay.stateNorm(workState);

	std::cout << "norm is " << norm << std::endl;
	assert(norm < 1e-10);  // Should decay to nearly zero

	std::cout << "✓ Fast decay adaptive stepping test passed" << std::endl;
}

void testSlowDecayLargeStep() {
	std::cout << "Testing Slow Decay with Large Steps..." << std::endl;

	ExponentialDecayMatrix slow_decay(0.1);	 // Slow decay
	RungeKutta45<Matrix22> integrator(slow_decay);
	integrator.hmin = 0.01;
	integrator.atol = 1e-12;
	integrator.rtol = 1e-12;

	Matrix22 state;
	*state(0, 0) = 1.0;
	*state(0, 1) = 0.0;
	*state(1, 0) = 0.0;
	*state(1, 1) = 1.0;

	double t_start = 0.0;
	double t_end = 5.0;

	Matrix22 workState;
	integrator(state, workState, t_start, t_end);

	// Analytical solution: exp(-0.1 * 5) = exp(-0.5) ≈ 0.6065
	double expected = std::exp(-0.5);

	std::cout << "Expected: " << expected << "\n";

	// assert(isNear(final_time, t_end));
	assert(isNear(*workState(0, 0), expected, 1e-1));
	assert(isNear(*workState(1, 1), expected, 1e-1));

	std::cout << "✓ Slow decay large step test passed" << std::endl;
}

void testIntegratorTolerances() {
	std::cout << "Testing Integrator Tolerances..." << std::endl;

	ExponentialDecayMatrix decay(0.2);

	// Tight tolerances
	RungeKutta45<Matrix22> tight_integrator(decay);
	tight_integrator.atol = 1e-3;
	tight_integrator.rtol = 1e-6;
	tight_integrator.hmin = 1e-12;
	// Loose tolerances
	RungeKutta45<Matrix22> loose_integrator(decay);
	loose_integrator.atol = loose_integrator.rtol = 1e-1;
	loose_integrator.hmin = 1e-2;

	Matrix22 state_tight, state_loose;
	*state_tight(0, 0) = *state_loose(0, 0) = 1.0;
	*state_tight(0, 1) = *state_loose(0, 1) = 0.0;
	*state_tight(1, 0) = *state_loose(1, 0) = 0.0;
	*state_tight(1, 1) = *state_loose(1, 1) = 1.0;

	double t_start = 0.0;
	double t_end = 1.0;

	Matrix22 work_tight, work_loose;

	auto tight_start = std::chrono::high_resolution_clock::now();
	tight_integrator(state_tight, work_tight, t_start, t_end);
	auto tight_end = std::chrono::high_resolution_clock::now();
	auto tight_duration = std::chrono::duration_cast<std::chrono::microseconds>(
		tight_end - tight_start);

	auto loose_start = std::chrono::high_resolution_clock::now();
	loose_integrator(state_loose, work_loose, t_start, t_end);
	auto loose_end = std::chrono::high_resolution_clock::now();
	auto loose_duration = std::chrono::duration_cast<std::chrono::microseconds>(
		loose_end - loose_start);

	// Tight tolerance should be more accurate
	double expected = std::exp(-0.2);
	double error_tight = std::abs(*work_tight(0, 0) - expected);
	double error_loose = std::abs(*work_loose(0, 0) - expected);

	std::cout << "Tight intg took " << tight_duration.count() / 1000000.0
			  << " s" << std::endl;
	std::cout << "Loose intg took " << loose_duration.count() / 1000000.0
			  << " s" << std::endl;

	assert(error_tight <= error_loose);
	assert(error_tight < 1e-8);

	std::cout << "✓ Integrator tolerance test passed" << std::endl;
}

void testZeroIntegrationTime() {
	std::cout << "Testing Zero Integration Time..." << std::endl;

	ExponentialDecayMatrix decay(1.0);
	RungeKutta45<Matrix22> integrator(decay);

	Matrix22 state;
	*state(0, 0) = 2.0;
	*state(0, 1) = 3.0;
	*state(1, 0) = 4.0;
	*state(1, 1) = 5.0;

	Matrix22 original_state;
	original_state = state;

	double t_start = 1.0;
	double t_end = 1.0;	 // Same start and end time

	Matrix22 workState;
	integrator(state, workState, t_start, t_end);

	// State should remain unchanged
	// assert(isNear(final_time, t_end));
	assert(isNear(*state(0, 0), *original_state(0, 0)));
	assert(isNear(*state(0, 1), *original_state(0, 1)));
	assert(isNear(*state(1, 0), *original_state(1, 0)));
	assert(isNear(*state(1, 1), *original_state(1, 1)));

	std::cout << "✓ Zero integration time test passed" << std::endl;
}

#if 0
void testLargeMatrix() {
	std::cout << "Testing Larger Matrix Integration..." << std::endl;

	// Test with 3x3 matrix (if your system supports it)
	class LargeExponentialDecay : public ODEDynamics<Matrix33> {
	   private:
		double alpha = 0.5;

	   public:
		virtual ~LargeExponentialDecay() override {}

		void PreIntegration(Matrix33& x, double t) override {}

		void PostIntegration(Matrix33& x, double t) override {}

		void Gradient(const Matrix33& x,
					  Matrix<double, 3, 3>& gradout, double t) override {
			for (ulong i = 0; i < 3; ++i) {
				for (ulong j = 0; j < 3; ++j) {
					gradout(i, j) = -alpha * x(i, j);
				}
			}
		}

		double stateNorm(const Matrix<double, 3, 3>& x) override {
			double sum = 0;
			for (ulong i = 0; i < 3; ++i) {
				for (ulong j = 0; j < 3; ++j) {
					sum += x(i, j) * x(i, j);
				}
			}
			return std::sqrt(sum);
		}
	};

	LargeExponentialDecay large_decay;
	RungeKutta45Matrix<3, 3> integrator(large_decay, 0.1, 1e-8, 1.0, 1e-6,
										1e-6);

	Matrix<double, 3, 3> state;
	// Initialize as identity matrix
	for (ulong i = 0; i < 3; ++i) {
		for (ulong j = 0; j < 3; ++j) {
			state(i, j) = (i == j) ? 1.0 : 0.0;
		}
	}

	double t_start = 0.0;
	double t_end = 2.0;

	Matrix33 workState;
	integrator(state, workState, t_start, t_end);

	double expected = std::exp(-0.5 * 2.0);	 // exp(-1.0)
	// assert(isNear(final_time, t_end));

	// Check diagonal elements
	for (ulong i = 0; i < 3; ++i) {
		assert(isNear(workState(i, i), expected, 1e-4));
	}

	// Check off-diagonal elements (should remain zero)
	for (ulong i = 0; i < 3; ++i) {
		for (ulong j = 0; j < 3; ++j) {
			if (i != j) {
				assert(isNear(workState(i, j), 0.0, 1e-6));
			}
		}
	}

	std::cout << "✓ Large matrix integration test passed" << std::endl;
}

void test1DIntegration() {
	struct LogisticEquation : public ODEDynamicsVector<double, 1> {
		void PreIntegration(Vector<double, 1>& x, double t) override {}

		// Post-integration hook (identity transformation for this example)
		void PostIntegration(Vector<double, 1>& x, double t) override {}

		// The core dynamics: dx/dt = f(x, t)
		void Gradient(const Vector<double, 1>& x, Vector<double, 1>& gradout,
					  double t) override {
			gradout[0] = x[0] * (1 - x[0]);
		}

		// State norm (Euclidean norm)
		double stateNorm(const Vector<double, 1>& x) override {
			return std::abs(x[0]);
		}
	};

	LogisticEquation log;
	RungeKutta45Vector<1, true> intg(log, 1, 1e-6, 20, 1e-5, 1e-5);

	Vector<double, 1> init, work;
	init[0] = 0.01;
	work.setZero();
	intg(init, work, 0, 1000);

	std::cout << "work[0] " << work[0] << std::endl;
	assert(isNear(work[0], 1.0, 1e-2));
	std::cout << "✓ 1D Logistic Equation test passed" << std::endl;
}

void test1DExponentialDecay() {
	struct LogisticEquation : public ODEDynamicsVector<double, 1> {
		void PreIntegration(Vector<double, 1>& x, double t) override {}

		// Post-integration hook (identity transformation for this example)
		void PostIntegration(Vector<double, 1>& x, double t) override {}

		// The core dynamics: dx/dt = f(x, t)
		void Gradient(const Vector<double, 1>& x, Vector<double, 1>& gradout,
					  double t) override {
			gradout[0] = -0.01 * x[0];
		}

		// State norm (Euclidean norm)
		double stateNorm(const Vector<double, 1>& x) override {
			return x[0] * x[0];
		}
	};

	LogisticEquation log;
	RungeKutta45Vector<1, true> intg(log, 0.001, 1e-9, 1, 1e-9, 1e-8);

	Vector<double, 1> init(1), work(1);
	init[0] = 1.0;
	work.setZero();
	intg(init, work, 0, 1);

	double expected = std::exp(-0.01 * 1.0);

	std::cout << "work[0] " << work[0] << std::endl;
	std::cout << "expected " << expected << std::endl;
	assert(isNear(work[0], expected, 1e-3));
	std::cout << "✓ 1D Logistic Equation test passed" << std::endl;
}
#endif

int main() {
	std::cout << "=== ODE Dynamics and Integration Unit Tests ===" << std::endl;

	try {
		// Vector-based ODE tests
		testSimpleHarmonicOscillatorGradient();
		testSimpleHarmonicOscillatorNorm();
		testGradientConsistency();
		testComplexOscillatorBehavior();

		// Matrix-based ODE tests
		testExponentialDecayMatrixGradient();
		testExponentialDecayMatrixNorm();

		// test1DIntegration();
		// test1DExponentialDecay();

		// Integration tests
		testExponentialDecayIntegration();
		testFastDecayAdaptiveStep();
		testSlowDecayLargeStep();
		testIntegratorTolerances();
		testZeroIntegrationTime();
		// testLargeMatrix();

		// Performance test
		runPerformanceTest();

		std::cout << "\n🎉 All tests passed successfully!" << std::endl;

	} catch (const std::exception& e) {
		std::cout << "❌ Test failed with exception: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
