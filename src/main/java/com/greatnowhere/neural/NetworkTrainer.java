package com.greatnowhere.neural;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Training 'manager' for a NN
 * Can resume training based on NNs serialized to file 
 * @author pzeltins
 *
 */
public class NetworkTrainer {

	List<TrainingSet> trainingSet = new ArrayList<TrainingSet>(); 
	double learningRate, momentum;
	/**
	 * network output that differs from expected by less than errorMargin will be considered 'success'
	 */
	public double errorMargin = 0.05D; 
	public int trainingIterations = 0;
	
	public Network n = new Network();
	
	public void init(int inputs, int outputs, int hidden, double learningRate, double momentum) {
		n.init(inputs, outputs, hidden);
		this.learningRate = learningRate;
		this.momentum = momentum;
	}
	
	public void init(String networkStateFilePath, double learningRate, double momentum) throws IOException {
		n = Network.fromPath(networkStateFilePath);
		this.learningRate = learningRate;
		this.momentum = momentum;
	}
	
	public void addTrainingSet(double[] inputs,double output,String label) {
		TrainingSet t = new TrainingSet();
		t.inputs = inputs;
		t.output = output;
		t.label = label;
		trainingSet.add(t);
	}
	
	
	
	public void merge(NetworkTrainer t) {
		this.n.merge(t.n);
	}

	public void train(String persistenceFile) throws InterruptedException, IOException {
		Utils.train(n, trainingSet, learningRate, momentum, errorMargin, persistenceFile);
	}

	public boolean trainCycle(String persistenceFile, int numIterations, int expectedSuccessRate, boolean printProgress) throws InterruptedException, IOException {
		return Utils.trainCycle(n, trainingSet, learningRate, momentum, errorMargin, persistenceFile, numIterations, expectedSuccessRate, printProgress);
	}
	
	public boolean trainGenetic(String persistenceFile, int numIterations, int expectedSuccessRate, int numSpecimens, int numGenerations, boolean printProgress) throws InterruptedException, IOException {
		n = Utils.trainGenetic(n, trainingSet,
				learningRate, momentum, errorMargin,
				persistenceFile, numSpecimens, numIterations, 
				numGenerations, expectedSuccessRate,printProgress);
		return n.successRate >= expectedSuccessRate;
	}
	
	public boolean trainHerd(String persistenceFile, int numIterations, int expectedSuccessRate, int numSpecimens, boolean printProgress) throws InterruptedException, IOException {
		n = Utils.trainHerd
				(n, trainingSet,learningRate, momentum, errorMargin,
				persistenceFile, numSpecimens, numIterations, 
				expectedSuccessRate,printProgress);
		return n.successRate >= expectedSuccessRate;
	}
	
}

/**
 * Class representing a single training set: inputs corresponding to output value
 * @author pzeltins
 *
 */
class TrainingSet implements Comparable<TrainingSet> {
	double[] inputs;
	double output;
	double error;
	String label;
	double order = r.nextDouble(); // random order
	
	static Random r = new Random();
	
	@Override
	// random sorting
	public int compareTo(TrainingSet arg0) {
		return Double.compare(order, arg0.order);
	}
	
}