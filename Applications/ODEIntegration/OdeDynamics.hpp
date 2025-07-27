#pragma once

#include <cstddef>

namespace swnumeric {

// Abstract class defining minimum interface for ODE Integration
template <typename StateType>
struct ODEDynamics {
	virtual ~ODEDynamics() = 0;
	virtual void PreIntegration(StateType& x, double t) = 0;
	virtual void PostIntegration(StateType& x, double t) = 0;
	virtual void Gradient(StateType& gradOut, const StateType& x, double t) = 0;
	virtual double stateNorm(const StateType& x) = 0;
};

template <typename StateType>
ODEDynamics<StateType>::~ODEDynamics(){};

};	// namespace swnumeric
