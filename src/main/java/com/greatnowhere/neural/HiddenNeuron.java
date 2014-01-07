package com.greatnowhere.neural;

/**
 * Defines a hidden neuron (one in middle layers of NN)
 * @author pzeltins
 *
 */
public class HiddenNeuron extends Neuron {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public HiddenNeuron(int mode, int alg) {
		super(mode,alg);
	}

	public double compute() {
		output = Utils.tanh(transferLinear());
		derivative = compute_derivative(output);
		return output;
	}

	@Override
	protected double compute_derivative(double y) {
		return Utils.d_tanh(y);
	}
	

}
