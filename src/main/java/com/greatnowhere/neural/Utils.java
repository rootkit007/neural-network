package com.greatnowhere.neural;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Utils {

	private static Random r = new Random();
	
	/**
	 * Sigmoid activation function
	 * @param x
	 * @return 0...1
	 */
	public static double sigmoid(double x) {
		if ( x < -45D ) return 0;
		if ( x > 45D ) return 1;
		return 1D / ( 1D + Math.exp(-x));
	}
	
	/**
	 * fast sigmoid approximation, MAD = 0.02
	 * @param x
	 * @return
	 */
	public static double sigmoid_approx(double x) {
		if ( x > 1D ) return 1;
		if ( x < -1D ) return 0;
		return 0.5 + x * (1 - Math.abs(x) / 2);
	}
	
	/**
	 * Computes sigmoid derivative
	 * @param y
	 * @return
	 */
	public static double d_sigmoid(double y) {
		return y * (1-y);
	}
	
	/**
	 * a simple sigmoid
	 * @param x
	 * @return -1...1
	 */
	public static double sigmoid_simple(double x) {
		return x / ( 1 + Math.abs(x));
	}
	
	public static double d_sigmoid_simple(double x) {
		return 1 / ( ( 1 + Math.abs(x)) * ( 1 + Math.abs(x)));
	}
	
	/**
	 * Tanh activation function
	 * @param x
	 * @return -1...1
	 */
	public static double tanh(double x) {
		//return (1-Math.exp(-2 * x)/(1+Math.exp(-2 * x)));
		return Math.tanh(x);
	}
	
	/**
	 * Tanh derivative
	 * @param y
	 * @return
	 */
	public static double d_tanh(double y) {
		return 1- (y * y);
	}
	
	
	public static double computeRpropUpdateCoeff(double previousUpdate,double currentUpdate) {
		switch ( (int)Math.signum(currentUpdate * previousUpdate) ) {
		case -1: // sign change
			return 0.5D; 
		case 1: // sign didnt change
			return 1.2D;
		default:
			return 1D;
		}
	}
	
	public static double getRpropUpdateValue(double updateValue, double prevUpdateValue,double currUpdateValue) {
		updateValue *= Utils.computeRpropUpdateCoeff(prevUpdateValue, currUpdateValue);
		updateValue = Math.min(Connection.MAX_RPROP_UPDATE_VALUE, Math.abs(updateValue)) * Math.signum(updateValue);
		updateValue = Math.max(Connection.MIN_RPROP_UPDATE_VALUE, Math.abs(updateValue)) * Math.signum(updateValue);
		return updateValue;
	}
	
	public static double train(Network n, List<TrainingSet> trainingSet,
			double learningRate, double momentum, double errorMargin,
			String persistenceFilePath) throws InterruptedException, IOException {
		n.maxError = 0;
		n.minError = Double.MAX_VALUE;
		n.currentIterations = 0;
		n.right = 0; n.wrong = 0;
		Collections.sort(trainingSet); //random shuffle
		n.totalError = 0D;
		for (TrainingSet t : trainingSet ) {
			t.order = TrainingSet.r.nextDouble();
			n.currentIterations++;
			n.setInputValues(t.inputs);
			n.compute();
			t.error = 0.5D * Math.pow( t.output - n.getOutput(), 2 ) ;
			n.learn(0, t.output, learningRate, momentum);
			if ( t.error < errorMargin ) n.right++; else n.wrong++;
			n.minError = ( t.error < n.minError ? t.error : n.minError );
			n.maxError = ( t.error > n.maxError ? t.error : n.maxError );
			n.totalError += t.error;
			n.squareError += Math.pow(t.error, 2);
			// System.out.print(getStats() + " error " + t.error + "\r");
		}
		n.commitChanges(learningRate,momentum);
		if ( persistenceFilePath != null )
			n.persistToFile(persistenceFilePath);
		n.successRate = new Double(n.right) / new Double(trainingSet.size()); 
		return n.successRate;
	}
	
	/**
	 * Runs training cycles until error falls below maxError, and number of training cycles are less than maxIterations
	 * Returns true if error fell below the threshold
	 * @param persistenceFilePath
	 * @param maxIterations 
	 * @param expected success rate
	 * @param printProgress
	 * @return
	 * @throws IOException 
	 * @throws InterruptedException 
	 */
	public static boolean trainCycle(Network n, List<TrainingSet> trainingSet,
			double learningRate, double momentum, double errorMargin,
			String persistenceFilePath, int maxIterations, double successRate,
			boolean printProgress) throws InterruptedException, IOException {
		
		int c=0;
		do {
			
			train(n, trainingSet, learningRate, momentum, errorMargin, persistenceFilePath);
			if ( printProgress ) {
				System.out.println("success rate " + n.successRate + " error " + n.totalError);
				System.out.println(n.toString());
			}
			
		} while ( n.successRate < successRate && c++ < maxIterations );

		return ( n.successRate >= successRate );
		
	}
	
	/**
	 * Genetic selection of networks
	 * @param persistenceFilePath
	 * @param numSpecimens
	 * @param maxIterations
	 * @param maxGenerations
	 * @param successRate
	 * @param printProgress
	 * @throws InterruptedException
	 * @throws IOException
	 */
	public static Network trainGenetic(Network n, List<TrainingSet> trainingSet,
			double learningRate, double momentum, double errorMargin,
			String persistenceFilePath, int numSpecimens, int maxIterations, 
			int maxGenerations, double successRate,	boolean printProgress) throws InterruptedException, IOException {
		
		ArrayList<Network> generation = new ArrayList<>();
		
		for (int g=0;g<maxGenerations;g++) {
			// train the best we got so far
			if ( trainCycle(n,trainingSet,learningRate,momentum,errorMargin,persistenceFilePath, maxIterations, successRate, printProgress) ) {
				break;
			}
			
			generation.clear();
			for ( int i=0; i < numSpecimens; i++) {
				Network specimen = new Network(n);
				// introduce random mutations
				mutateNetwork(specimen); mutateNetwork(specimen);
				generation.add(specimen);
				if ( trainCycle(specimen,trainingSet,learningRate,momentum,errorMargin,persistenceFilePath, maxIterations, successRate, printProgress) )
					break;
			}
			// first item should be be best network
			Collections.sort(generation);
			Network survivor = generation.get(0);
			if ( survivor.successRate >= successRate ) {
				n = survivor;
				break;
			}
			
		}
		
		return n;
		
	}
	
	/**
	 * Creates numSpecimens networks, trains each separately, and chooses the best one
	 * @param n
	 * @param trainingSet
	 * @param learningRate
	 * @param momentum
	 * @param errorMargin
	 * @param persistenceFilePath
	 * @param numSpecimens
	 * @param maxIterations
	 * @param successRate
	 * @param printProgress
	 * @return
	 * @throws InterruptedException
	 * @throws IOException
	 */
	public static Network trainHerd(Network n, List<TrainingSet> trainingSet,
			double learningRate, double momentum, double errorMargin,
			String persistenceFilePath, int numSpecimens, int maxIterations, 
			double successRate,	boolean printProgress) throws InterruptedException, IOException {
		
		ArrayList<Network> herd = new ArrayList<>();
		
		// train what we got so far
		if ( trainCycle(n,trainingSet,learningRate,momentum,errorMargin,persistenceFilePath, maxIterations, successRate, printProgress) ) {
			return n;
		}
			
		herd.clear();
		for ( int i=0; i < numSpecimens; i++) {
			Network specimen = new Network(n);
			herd.add(specimen);
			if ( trainCycle(specimen,trainingSet,learningRate,momentum,errorMargin,persistenceFilePath, maxIterations, successRate, printProgress) )
				break;
		}
		// first item should be be best network
		Collections.sort(herd);
		n = herd.get(0);
			
		return n;
		
	}
	
	/**
	 * Randomly mutates said network by adding a connection or a neuron
	 * @param n
	 */
	public static void mutateNetwork(Network n) {
		switch (r.nextInt(2)) {
		case 0: // add connection
			switch (r.nextInt(4)) {
			case 0: // connect input with hidden
				n.hidden.get(r.nextInt(n.hiddenLayerSize)).connectInput(n.inputValues[r.nextInt(n.inputLayerSize)], null);
				break;
			case 1: // connect two hiddens
				// get one hidden with layer at least 0
				HiddenNeuron n1 = findNeuron(n, 0, Neuron.LAYER_OUTPUT);
				if ( n1 != null ) {
					HiddenNeuron n2 = findNeuron(n, n1.layer+1, Neuron.LAYER_OUTPUT);
					if ( n2 != null ) {
						n1.connectOutputNeuron(n2, null);
					}
				}
				
				break;
			case 2: // connect hidden with output
				n.hidden.get(r.nextInt(n.hiddenLayerSize)).connectOutputNeuron(n.output.get(r.nextInt(n.outputLayerSize)),null);
				break;
			case 3: // connect input to output
				n.output.get(r.nextInt(n.outputLayerSize)).connectInput(n.inputValues[r.nextInt(n.inputLayerSize)], null);
				break;
			}
			break;
		case 1: // add neuron
			HiddenNeuron hn = n.addHiddenNeuron();
			// and a random connection to and from said neuron
			switch (r.nextInt(3)) {
			case 0: // connect to input and output 
				hn.connectInput(n.inputValues[r.nextInt(n.inputLayerSize)], null);
				hn.connectOutputNeuron(n.output.get(r.nextInt(n.outputLayerSize)),null);
				break;
			case 1: // connect to hidden and hidden
				// get one hidden with layer at least 0
				HiddenNeuron n1 = findNeuron(n, 0, Neuron.LAYER_OUTPUT);
				if ( n1 != null ) {
					HiddenNeuron n2 = findNeuron(n, n1.layer+2, Neuron.LAYER_OUTPUT);
					if ( n2 != null ) {
						hn.connectInput(n1, null);
						hn.connectOutputNeuron(n2, null);
						hn.layer = n1.layer + 1;
					}
				}
				
				break;
			case 2: // connect to hidden and output 
				HiddenNeuron hn1 = findNeuron(n, 0, Neuron.LAYER_OUTPUT);
				if ( hn1 != null ) {
					hn.connectInput(hn1, null);
					hn.connectOutputNeuron(n.output.get(r.nextInt(n.outputLayerSize)),null);
					hn.layer = hn1.layer + 1;
				}
				break;
			}
			break;
		}
	}
	
	/**
	 * finds and returns random hidden neuron in the network where layer>=minLayer and layer<maxLayer
	 * @param n
	 * @param minLayer
	 * @param maxLayer
	 * @return null if cannot be found
	 */
	public static HiddenNeuron findNeuron(Network net, int minLayer, int maxLayer) {
		ArrayList<HiddenNeuron> candidates = new ArrayList<>();
		for (HiddenNeuron n : net.hidden ) {
			if ( n.layer >= minLayer && n.layer < maxLayer ) 
				candidates.add(n);
		}
		if ( candidates.size() == 0 ) return null;
		return candidates.get(r.nextInt(candidates.size()));
	}
}
