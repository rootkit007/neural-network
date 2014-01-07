package com.greatnowhere.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;

/**
 * Abstract base class for neuron
 * @author pzeltins
 *
 */
public abstract class Neuron implements IOutput,Runnable, Serializable, IBatchLearning {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	transient private CountDownLatch latch;
	
	public static final int TRAINING_MODE_STOCHASTIC = 1; // weights updated after each training set
	public static final int TRAINING_MODE_BATCH = 2; // weights updated after all training sets are run
	public static final int TRAINING_MODE_BATCH_GRADIENT = 3; // weights updated calculated once after all training sets are run 
	int trainingMode = TRAINING_MODE_STOCHASTIC;
	
	public static final int TRAINING_ALG_BACKPROP = 1; // classic backpropagation
	public static final int TRAINING_ALG_RPROP = 2; // RPROP backpropagation
	int trainingAlgorithm = TRAINING_ALG_BACKPROP;
	
	public static final int LAYER_INPUT = 0;
	public static final int LAYER_OUTPUT = Integer.MAX_VALUE;
	/**
	 * layer number this is in
	 */
	int layer = LAYER_INPUT; 
	
	public Neuron() {
		this.connectInput(new BiasConnection());
	}
	
	public Neuron(int trainingMode,int trainingAlgorithm) {
		this();
		this.trainingMode = trainingMode;
		this.trainingAlgorithm = trainingAlgorithm;
	}
	
	double gradient = 0D;
	double batchGradient = 0D;
	double prevGradient = 0D;
	
	static Random r = new Random();
	
	ArrayList<Connection> outputConnections = new ArrayList<Connection>();
	ArrayList<Connection> inputConnections = new ArrayList<Connection>();
	
	double output;
	double derivative;
	
	int numInputs;
	
	public abstract double compute();
	protected abstract double compute_derivative(double y);
	
	public void computeAsync(CountDownLatch latch, Executor exec) {
		this.latch = latch;
		exec.execute(this);
	}
	
	double transferLinear() {
		// linear transfer function
		double t = 0D;
		for (Connection c : inputConnections ) {
			t += c.source.getOutput() * c.destWeight;
		}
		return t;
	}
	
	boolean connectOutputNeuron(Neuron n,Connection c) {
		c = ( c == null ? new Connection() : c);
		c.source = this;
		c.dest = n;
		if ( !n.inputConnections.contains(c) ) {
			outputConnections.add(c);
			n.inputConnections.add(c);
			return true;
		}
		return false;
	}

	/**
	 * Connects this neuron to specified output
	 * @param n
	 * @param c, can be null
	 * @return true if conencted, false if error occured (eg connection already exists)
	 */
	boolean connectInput(IOutput n, Connection c) {
		c = ( c == null ? new Connection() : c);
		c.source = n;
		c.dest = this;
		if ( !inputConnections.contains(n) ) { 
			inputConnections.add(c);
			if ( n instanceof Neuron ) {
				((Neuron) n).outputConnections.add(c);
			}
			return true;
		}
		return false;
	}
	
	void connectInput(Connection c) {
		c.dest = this;
		inputConnections.add(c);
		if ( c.source instanceof Neuron ) {
			((Neuron) c.source).outputConnections.add(c);
		}
	}
	
	/**
	 * applicable to output neurons only
	 * @param target value
	 * @param learningRate. not used if algorithm is RPROP
	 * @param momentum. not used if algorithm is RPROP
	 */
	void learn(double target, double learningRate, double momentum) {
		gradient = ( target - output ) * derivative;
		batchGradient += gradient;
		adjustInputWeights(learningRate, momentum);
	}
	
	
	/**
	 * applicable to hidden layers neurons only
	 * @param learningRate. not used if algorithm is RPROP
	 * @param momentum. not used if algorithm is RPROP
	 */
	void learn(double learningRate,double momentum) {
		double sumOutputGradients = 0D;
		for ( Connection c : outputConnections ) {
			sumOutputGradients += c.dest.gradient * c.destWeight;
		}
		gradient = sumOutputGradients * derivative;
		batchGradient += gradient;
		adjustInputWeights(learningRate, momentum);
	}
	
	protected void adjustInputWeights(double learningRate, double momentum) {
		// no adjustments in batch gradient mode - we only need total gradient for batch
		if ( this.trainingMode == TRAINING_MODE_BATCH_GRADIENT ) 
			return;
		
		double k = learningRate * gradient;
		for (Connection c : inputConnections ) {
			// http://en.wikipedia.org/wiki/Backpropagation
			if ( c.source != null ) {
				double deltaW = 0D;
				if ( this.trainingAlgorithm == TRAINING_ALG_BACKPROP ) { 
					// compute weight delta for backprop
					deltaW = k * c.source.getOutput() + c.deltaWeight * momentum;
				} else {
					c.prevDeltaWeight = c.updateValue;
					deltaW = Utils.getRpropUpdateValue(c.updateValue, c.prevDeltaWeight, gradient);
					c.updateValue = deltaW;
				}
				c.deltaWeight = deltaW;
				c.totalDeltaWeight += deltaW; // accumulate total delta weight for batch run
				if ( trainingMode == TRAINING_MODE_STOCHASTIC )
					c.destWeight += deltaW;
			}
		}
	}

	public double getOutput() {
		return output;
	}

	@Override
	public void run() {
		compute();
		if ( latch != null ) {
			latch.countDown();
		}
	}
	
	public void merge(Neuron n) {
		for (int i=0; i<this.inputConnections.size(); i++) {
			Connection c = inputConnections.get(i);
			Connection c1 = n.inputConnections.get(i);
			c.merge(c1);
		}
	}

	@Override
	public void commitWeights(double learningRate, double momentum) {
		if ( trainingMode != TRAINING_MODE_STOCHASTIC ) {
			double k = learningRate * batchGradient;
			for (Connection c : inputConnections ) {
				if ( c.source != null ) {
					double deltaW = 0D;
					if ( this.trainingAlgorithm == TRAINING_ALG_BACKPROP ) { 
						// compute weight delta for backprop
						deltaW = k * c.source.getOutput() + c.deltaWeight * momentum;
					} else {
						if ( trainingMode == TRAINING_MODE_BATCH_GRADIENT ) {
							deltaW = Utils.getRpropUpdateValue(c.updateValue, c.prevDeltaWeight, batchGradient);
							c.prevDeltaWeight = c.updateValue;
							c.updateValue = deltaW;
						} else {
							deltaW = c.totalDeltaWeight;
						}
					}
					c.deltaWeight = deltaW;
					c.destWeight += deltaW;
				}
				c.totalDeltaWeight = 0D;
			}
		}
		batchGradient = 0D;
	}

}
