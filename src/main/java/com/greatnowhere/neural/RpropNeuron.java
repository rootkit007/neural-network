package com.greatnowhere.neural;

/**
 * Class representing neuron with RPROP learning method
 * @author pzeltins
 *
 */
public abstract class RpropNeuron extends Neuron {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	double previousGradient;
	
	void learn(double target, double learningRate, double momentum) {
		gradient = ( target - output ) * derivative;
		adjustInputWeightsRprop();
	}
	
	
	void learn(double learningRate,double momentum) {
		double sumOutputGradients = 0D;
		for ( Connection c : outputConnections ) {
			sumOutputGradients += c.dest.gradient * c.destWeight;
		}
		gradient = sumOutputGradients * derivative;
		adjustInputWeightsRprop();
	}
	
	private void adjustInputWeightsRprop() {
		for (Connection c : inputConnections ) {
			if ( c.source != null ) {
				c.totalDeltaWeight += gradient * c.source.getOutput();
			}
		}
	}

}
