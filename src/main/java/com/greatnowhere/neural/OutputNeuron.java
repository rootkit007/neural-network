package com.greatnowhere.neural;

/**
 * Class representing output neuron with sigmoid activation func
 * @author pzeltins
 *
 */
public class OutputNeuron extends Neuron {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public OutputNeuron(int mode, int alg) {
		super(mode,alg);
		layer = LAYER_OUTPUT;
	}

	public double compute() {
		output = Utils.sigmoid(transferLinear());
		derivative = compute_derivative(output);
		return output;
	}
	
	@Override
	protected double compute_derivative(double y) {
		return Utils.d_sigmoid(y);
	}
	
}
