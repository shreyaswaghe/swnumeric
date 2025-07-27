#pragma once

#include <algorithm>
#include <vector>

#include "Libraries/Tensor/Tensor.hpp"
#include "OdeDynamics.hpp"

namespace swnumeric {

struct DormandPrinceTableau {
	// clang-format off
	static constexpr double b[7] = {  
		35.0000 / 384.0,				  
		0.000,						  
		500.000 / 1113.0,				  
		125.000 / 192.0,				  
		-2187.0 / 6784.0,			  
		11.0000 / 84.0,				  
		0.00000};
	static constexpr double bstar[7] = {  
		5179.0 / 57600.0,				  
		0.0,							  
		7571.0 / 16695.0,				  
		393.0 / 640.0,					  
		-92097.0 / 339200.0,			  
		187.0 / 2100.0,					  
		1.0 / 40.0};

	static constexpr double bdiff[7] = {
		b[0] - bstar[0],
    b[1] - bstar[1],
    b[2] - bstar[2],
    b[3] - bstar[3],
		b[4] - bstar[4],
    b[5] - bstar[5],
    b[6] - bstar[6]
  };

	static constexpr double c[7] = {
    0.0,
    1.0 / 5.0,
    3.0 / 10.0,
    4.0 / 5.0,
    8.0 / 9.0,
    1.0,
    1.0};

	static constexpr double a[7][6] = {
		{0.0, 0.0, 0.0,0.0,0.0,0.0},
		{1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
		{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0},
		{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
		 0.0, 0.0},
		{9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
		 -5103.0 / 18656.0, 0.0},
		{35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
		 11.0 / 84.0}};
};

struct CashKarpTableau {
	static constexpr double b[6] = {  
		37.0 / 378.0,				  
		0.0,						  
		250.0 / 621.0,				  
		125.0 / 594.0,				  
		0.0,						  
		512.0 / 1771.0};
	static constexpr double bstar[6] = {  
		2825.0 / 27648.0,				  
		0.0,							  
		18575.0 / 48384.0,				  
		13525.0 / 55296.0,				  
		277.0 / 14336.0,				  
		1.0 / 4.0};
	static constexpr double bdiff[6] = {b[0] - bstar[0], b[1] - bstar[1],
										b[2] - bstar[2], b[3] - bstar[3],
										b[4] - bstar[4], b[5] - bstar[5]};
	static constexpr double c[6] = {0.0,	   1.0 / 5.0, 3.0 / 10.0,
									3.0 / 5.0, 1.0,		  7.0 / 8.0};
	static constexpr double a[6][5] = {
		{0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0 / 5.0, 0.0, 0.0, 0.0, 0.0},
		{3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0},
		{3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0},
		{-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0},
		{1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0,
		 253.0 / 4096.0}};
	// clang-format on
};

template <typename StateType>
struct RungeKutta45 {
	static_assert(std::is_base_of<TensorLike, StateType>::value,
				  "YOU CAN ONLY OPERATE ON TENSORS!!! RK45");

	ODEDynamics<StateType>& ode;
	using Tableau = CashKarpTableau;

	StateType k1;
	StateType k2;
	StateType k3;
	StateType k4;
	StateType k5;
	StateType k6;
	StateType kdiff;

	StateType l1;
	StateType l2;
	StateType l3;
	StateType l4;
	StateType l5;
	StateType l6;

	std::vector<uint32_t> numLoopEvals;
	std::vector<float> timeSteps;

	double h = 1;
	double hmin = 1e-6, hmax = 1e2;
	double atol = 1e-6, rtol = 1e-6;

	char maxIter = 48;
	bool recordStats = false;

	RungeKutta45(ODEDynamics<StateType>& ode, bool recordStats = false)
		: ode(ode), recordStats(recordStats) {};

	void operator()(const StateType& state, StateType& workState,
					const double t, const double t_end) {
		workState = state;
		double time = t;
		while (time < t_end) {
			ode.PreIntegration(workState, time);
			time += step(workState, time, t_end);
			ode.PostIntegration(workState, time);
		}
	}

	double step(StateType& state, const double t, const double t_end) {
		// alloc only happens on first invocation
		k1.alloc(state.shape());
		k2.alloc(state.shape());
		k3.alloc(state.shape());
		k4.alloc(state.shape());
		k5.alloc(state.shape());
		k6.alloc(state.shape());
		kdiff.alloc(state.shape());

		l1.alloc(state.shape());
		l2.alloc(state.shape());
		l3.alloc(state.shape());
		l4.alloc(state.shape());
		l5.alloc(state.shape());
		l6.alloc(state.shape());

		double stateNorm = ode.stateNorm(state);

		double h = this->h;
		double hprop = 0.0;

		const uint8_t maxIters = maxIter;
		uint8_t loopEvals = 0;
		for (; loopEvals < maxIters; loopEvals++) {
			double err_estimate = 0.0;

			// this isn't particularly necessary but good to start on a clean
			// slate
			k1.setZero();
			k2.setZero();
			k3.setZero();
			k4.setZero();
			k5.setZero();
			k6.setZero();

			// copy state over
			l1 = state;
			l2 = state;
			l3 = state;
			l4 = state;
			l5 = state;
			l6 = state;

			ode.Gradient(k1, l1, t);
			k1 *= h;

			l2 += Tableau::a[1][0] * k1;
			ode.Gradient(k2, l2, t + h * Tableau::c[1]);
			k2 *= h;

			l3 += Tableau::a[2][0] * k1;
			l3 += Tableau::a[2][1] * k2;
			ode.Gradient(k3, l3, t + h * Tableau::c[2]);
			k3 *= h;

			l4 += Tableau::a[3][0] * k1;
			l4 += Tableau::a[3][1] * k2;
			l4 += Tableau::a[3][2] * k3;
			ode.Gradient(k4, l4, t + h * Tableau::c[3]);
			k4 *= h;

			l5 += Tableau::a[4][0] * k1;
			l5 += Tableau::a[4][1] * k2;
			l5 += Tableau::a[4][2] * k3;
			l5 += Tableau::a[4][3] * k4;
			ode.Gradient(k5, l5, t + h * Tableau::c[4]);
			k5 *= h;

			l6 += Tableau::a[5][0] * k1;
			l6 += Tableau::a[5][1] * k2;
			l6 += Tableau::a[5][2] * k3;
			l6 += Tableau::a[5][3] * k4;
			l6 += Tableau::a[5][4] * k5;
			ode.Gradient(k6, l6, t + h * Tableau::c[5]);
			k6 *= h;

			// the difference in updates will be computed to k6
			kdiff.setZero();
			kdiff += Tableau::bdiff[0] * k1;
			kdiff += Tableau::bdiff[1] * k2;
			kdiff += Tableau::bdiff[2] * k3;
			kdiff += Tableau::bdiff[3] * k4;
			kdiff += Tableau::bdiff[4] * k5;
			kdiff += Tableau::bdiff[5] * k6;

			err_estimate = ode.stateNorm(kdiff);
			err_estimate /= atol + rtol * stateNorm;

			// compute proposed update timestep
			hprop = 0.97 * h * std::pow(err_estimate, -0.20);
			h = (h >= 0.0 ? std::max(hprop, 0.05 * h)
						  : std::min(hprop, 20 * h));
			h = std::max(h, hmin);

			if (err_estimate <= 1.0 || h == hmin) {
				break;
			}
		}

		// if the step suggested is out of bounds, clip it to bounds and make
		// another update step calculation
		if (h > hmax || (t + h) > t_end) {
			loopEvals++;
			h = std::min(hmax, t_end - t);

			ode.Gradient(l1, k1, t);
			k1 *= h;

			l2 += Tableau::a[1][0] * k1;
			ode.Gradient(l2, k2, t + h * Tableau::c[1]);
			k2 *= h;

			l3 += Tableau::a[2][0] * k1;
			l3 += Tableau::a[2][1] * k2;
			ode.Gradient(l3, k3, t + h * Tableau::c[2]);
			k3 *= h;

			l4 += Tableau::a[3][0] * k1;
			l4 += Tableau::a[3][1] * k2;
			l4 += Tableau::a[3][2] * k3;
			ode.Gradient(l4, k4, t + h * Tableau::c[3]);
			k4 *= h;

			l5 += Tableau::a[4][0] * k1;
			l5 += Tableau::a[4][1] * k2;
			l5 += Tableau::a[4][2] * k3;
			l5 += Tableau::a[4][3] * k4;
			ode.Gradient(l5, k5, t + h * Tableau::c[4]);
			k5 *= h;

			l6 += Tableau::a[5][0] * k1;
			l6 += Tableau::a[5][1] * k2;
			l6 += Tableau::a[5][2] * k3;
			l6 += Tableau::a[5][3] * k4;
			l6 += Tableau::a[5][4] * k5;
			ode.Gradient(l6, k6, t + h * Tableau::c[5]);
			k6 *= h;
		}

		// k1 is now the full update step
		k1 *= Tableau::b[0];
		k1 += Tableau::b[1] * k2;
		k1 += Tableau::b[2] * k3;
		k1 += Tableau::b[3] * k4;
		k1 += Tableau::b[4] * k5;
		k1 += Tableau::b[5] * k6;

		// record stats if requested
		if (recordStats) {
			numLoopEvals.push_back(loopEvals);
			timeSteps.push_back(h);
		}

		// store integrator state
		this->h = h;

		// update state variable
		state += k1;

		// return timestep
		return h;
	}
};

};	// namespace swnumeric
