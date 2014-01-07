package com.greatnowhere.neural;

/**
 * Represents an input value to NN
 * @author pzeltins
 *
 */
public class InputValue implements IOutput {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public double value = 0D;
	
	public double getOutput() {
		return value;
	}

}
