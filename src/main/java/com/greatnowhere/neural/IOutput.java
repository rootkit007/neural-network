package com.greatnowhere.neural;

import java.io.Serializable;

/**
 * Interface describing object that provides an output value (eg neuron, input to the network etc) 
 * @author pzeltins
 *
 */
public interface IOutput extends Serializable {

	public double getOutput();
	
}
