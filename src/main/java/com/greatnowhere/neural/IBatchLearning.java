package com.greatnowhere.neural;

/**
 * Interface that defines batch learning behaviour (ie where weights are adjusted only after full round of learning
 * @author pzeltins
 *
 */
public interface IBatchLearning {

	public void commitWeights(double learningRate, double momentum);
	
}
