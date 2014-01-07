package com.greatnowhere.neural;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Class representing 3-layer NN  
 * @author pzeltins
 *
 */
public class Network implements Serializable, Comparable<Network> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	int inputLayerSize; // number of inputs
	int outputLayerSize; // number of outputs
	int hiddenLayerSize;
	
	List<Double> hiddenWeights; // used only for serializing/deserialzing
	List<Double> outputWeights;
	
	private int trainingMethod = Neuron.TRAINING_MODE_STOCHASTIC; 
	private int trainingAlgorithm = Neuron.TRAINING_ALG_BACKPROP;
	
	public double error=0D;
	public double minError = Double.MAX_VALUE, stdError = 0D, totalError = 0D, maxError = 0D, squareError = 0D;
	public double successRate = 0D;
	public int right, wrong;

	/**
	 * Number of iterations processed in current trainig set batch
	 */
	public int currentIterations = 0;

	/**
	 * The following properties are transient so whole NN can be serialized without taking up too much space
	 */
	transient ExecutorService executor;
	transient InputValue[] inputValues;
	transient ArrayList<HiddenNeuron> hidden;
	transient ArrayList<OutputNeuron> output;
	
	public Network() {
		
	}
	
	/**
	 * Create new network with the same dimensions and parameters as source
	 * Does not copy weights
	 * @param source
	 */
	public Network(Network source) {
		this();
		inputLayerSize = source.inputLayerSize;
		outputLayerSize = source.outputLayerSize;
		hiddenLayerSize = source.hiddenLayerSize;
		this.hidden = new ArrayList<HiddenNeuron>();
		this.output = new ArrayList<OutputNeuron>();
		executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()+1);
		
		// input layer
		inputValues = new InputValue[inputLayerSize];
		for (int i=0; i<inputLayerSize; i++) {
			inputValues[i] = new InputValue();
		}
		// hidden layer
		for (int i=0;i<hiddenLayerSize;i++) {
			HiddenNeuron n = new HiddenNeuron(trainingMethod,trainingAlgorithm);
			this.hidden.add(n);
		}
		for (int i=0;i<outputLayerSize;i++) {
			OutputNeuron n = new OutputNeuron(trainingMethod,trainingAlgorithm);
			this.output.add(n);
		}
		// duplicate connections (but not weights)
		List<InputValue> sourceNetInputs = Arrays.asList(source.inputValues);
		for (int i=0;i<hiddenLayerSize;i++) {
			HiddenNeuron n = hidden.get(i);
			for (Connection c : source.hidden.get(i).inputConnections ) {
				if ( c.source instanceof InputValue ) {
					int inputIdx = sourceNetInputs.indexOf(c.source);
					if ( inputIdx > -1 ) {
						n.connectInput(inputValues[inputIdx], null);
					}
				} else {
					int inputIdx = source.hidden.indexOf(c.source);
					n.connectInput(hidden.get(inputIdx), null);
				}
			}
			for (Connection c : source.hidden.get(i).outputConnections ) {
				int inputIdx = sourceNetInputs.indexOf(c.source);
				if ( inputIdx > -1 ) {
					n.connectInput(inputValues[inputIdx], null);
				}
			}
				
		}
		setTrainingAlgorithm(source.getTrainingAlgorithm());
		setTrainingMethod(source.getTrainingMethod());
	}
	
	/**
	 * Initializes NN to the specified dimensions
	 * @param inputs
	 * @param outputs
	 * @param hidden
	 */
	public void init(int inputs,int outputs,int hidden) {
		inputLayerSize = inputs;
		outputLayerSize = outputs;
		hiddenLayerSize = hidden;
		this.hidden = new ArrayList<HiddenNeuron>();
		this.output = new ArrayList<OutputNeuron>();
		// by default network will be calculated by an executor having cores+1 threads
		executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()+1);
		
		// input layer
		inputValues = new InputValue[inputs];
		for (int i=0; i<inputs; i++) {
			inputValues[i] = new InputValue();
		}
		
		// hidden layer
		for (int i=0;i<hiddenLayerSize;i++) {
			HiddenNeuron n = new HiddenNeuron(trainingMethod,trainingAlgorithm);
			// connect every hidden layer neuron to every input layer neuron
			for (int j=0; j<inputs; j++) {
				n.connectInput(inputValues[j], null);
			}
			this.hidden.add(n);
		}

		
		// output layer
		for (int i=0;i<outputLayerSize;i++) {
			OutputNeuron n = new OutputNeuron(trainingMethod,trainingAlgorithm);
			// connect every hidden layer neuron to every output layer neuron
			for (Neuron nInp : this.hidden ) {
				n.connectInput(nInp, null);
			}
			this.output.add(n);
		}
	}
	
	/**
	 * Stores weights into hiddenWeights and outputWeights lists
	 * so this NN can be efficiently serialized
	 */
	public void storeWeights() {
		ArrayList<Double> r = new ArrayList<>();
		for (Neuron n : hidden ) {
			for ( Connection c : n.inputConnections ) {
				r.add(c.destWeight);
			}
		}
		hiddenWeights = r;
		r = new ArrayList<>();
		for (Neuron n : output ) {
			for ( Connection c : n.inputConnections ) {
				r.add(c.destWeight);
			}
		}
		outputWeights = r;
	}
	
	/**
	 * Restores full connections graph from hiddenWeights and outputWeights lists
	 * Used after deserializing
	 */
	public void restoreWeights() {
		int i=0;
		for (Neuron n : hidden ) {
			for ( Connection c : n.inputConnections ) {
				c.destWeight = hiddenWeights.get(i++);
			}
		}
		i=0;
		for (Neuron n : output ) {
			for ( Connection c : n.inputConnections ) {
				c.destWeight = outputWeights.get(i++);
			}
		}
	}
	
	/**
	 * Used in genetic algorithm to add a hidden neuron(s) to NN
	 * @param count
	 */
	public void addHiddenNeuron(int count) {
		for (int i=0;i<count;i++) {
			HiddenNeuron n = new HiddenNeuron(trainingMethod,trainingAlgorithm);
			// connect every hidden layer neuron to every input layer neuron
			for (int j=0; j<this.inputLayerSize; j++) {
				n.connectInput(inputValues[j], null);
			}
			// connect to output layer
			for (int j=0; j<this.outputLayerSize; j++) {
				Neuron o = this.output.get(j);
				o.connectInput(n, null);
			}
			this.hidden.add(n);
		}
		this.hiddenLayerSize += count;
	}
	
	public HiddenNeuron addHiddenNeuron() {
		HiddenNeuron n = new HiddenNeuron(trainingMethod,trainingAlgorithm);
		this.hidden.add(n);
		this.hiddenLayerSize ++;
		return n;
	}
	
	/**
	 * Serializes this NN and returns base64-encoded string representation
	 * @return
	 */
	public String serialize() {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		try {
	        ObjectOutputStream oos = new ObjectOutputStream( baos );
	        storeWeights();
	        oos.writeObject( this );
	        oos.close();
		} catch (IOException ex) {
			
		}
        return new String( Base64Coder.encode( baos.toByteArray() ) );
	}
	
	/**
	 * Serializes and writes NN to file
	 * @param filepath
	 * @throws IOException
	 */
	public void persistToFile(String filepath) throws IOException {
		Files.write(Paths.get(filepath), serialize().getBytes(StandardCharsets.UTF_8));
	}
	
	/**
	 * Deserializes NN from base64-encoded string 
	 * @param s
	 * @return
	 */
	public static Network fromString(String s) {
		try {
			byte [] data = Base64Coder.decode( s );
	        ObjectInputStream ois = new ObjectInputStream( 
	                                        new ByteArrayInputStream( data ) );
	        Network o  = (Network) ois.readObject();
	        ois.close();
	        o.init(o.inputLayerSize, o.outputLayerSize, o.hiddenLayerSize);
	        o.restoreWeights();
	        return o;
		} catch (IOException|ClassNotFoundException e) {
			return null;
		}
	}
	
	/**
	 * Deserializes NN from file
	 * @param filePath
	 * @return
	 * @throws IOException
	 */
	public static Network fromPath(String filePath) throws IOException {
		byte[] encoded = Files.readAllBytes(Paths.get(filePath));
		Network n = fromString(StandardCharsets.UTF_8.decode(ByteBuffer.wrap(encoded)).toString());
		n.executor = Executors.newFixedThreadPool(20);
		return n;
	}
	
	public void setInputValues(double[] v) {
		for (int i=0; i<inputLayerSize; i++) {
			inputValues[i].value = v[i];
		}
	}
	
	/**
	 * Computes NN using multiple-threaded executor
	 * @throws InterruptedException
	 */
	public void compute() throws InterruptedException {
		// forward pass computing, establish a latch that will allow us to wait until whole hidden layer is computed
		CountDownLatch latch;
		latch = new CountDownLatch(this.hiddenLayerSize);
		for (Neuron n : this.hidden ) {
			n.computeAsync(latch, this.executor);
		}
		latch.await();
		// latch for output layer
		latch = new CountDownLatch(this.outputLayerSize);
		for (Neuron n : this.output ) {
			n.computeAsync(latch, this.executor);
		}
		latch.await();
	}
	
	/**
	 * Returns value of output neuron n
	 * @param n
	 * @return
	 */
	public double getOutput(int n) {
		return this.output.get(n).getOutput();
	}
	
	public double getOutput() {
		return getOutput(0);
	}
	
	/**
	 * Executes single backpropagatio pass
	 * @param outputIndex
	 * @param target
	 * @param learningRate
	 * @param momentum
	 */
	public void learn(int outputIndex, double target, double learningRate, double momentum) {
		// backpropagation pass
		this.output.get(outputIndex).learn(target, learningRate, momentum);
		for (Neuron n : this.hidden ) {
			n.learn(learningRate, momentum);
		}
	}
	
	/**
	 * Used for batch learning - commits changes to weights 
	 * @param learningRate
	 * @param momentum
	 */
	public void commitChanges(double learningRate, double momentum) {
		for (Neuron n : this.output ) {
			if ( n instanceof IBatchLearning )
				((IBatchLearning) n).commitWeights(learningRate,momentum);
		}
		for (Neuron n : this.hidden ) {
			if ( n instanceof IBatchLearning )
				((IBatchLearning) n).commitWeights(learningRate,momentum);
		}
	}
	
	/**
	 * Merges this NN with another. Both NNs must have the same dimensions
	 * @param net
	 */
	public void merge(Network net) {
		for (int i=0; i<this.hidden.size(); i++) {
			Neuron n = this.hidden.get(i);
			n.merge(net.hidden.get(i));
		}
		for (int i=0; i<this.output.size(); i++) {
			Neuron n = this.output.get(i);
			n.merge(net.output.get(i));
		}
	}
	
	public String toString() {
		String s = "";
		int i=0;
		for ( Neuron n : this.hidden) {
			s += "(" + ++i + ")";
			for ( Connection c : n.inputConnections ) {
				s += c.toString();
			}
		}
		s += "\n";
		i=0;
		for ( Neuron n : this.output) {
			s += "(" + ++i + ")";
			for ( Connection c : n.inputConnections ) {
				s += c.toString();
			}
		}
		s += "\n";
		return s;
	}

	/**
	 * Training method, see Neuron 
	 * 
	 * @return
	 */
	public int getTrainingMethod() {
		return trainingMethod;
	}

	/**
	 * Applies training method to all neurons in the network
	 * @param trainingMethod
	 */
	public void setTrainingMethod(int trainingMethod) {
		this.trainingMethod = trainingMethod;
		for (Neuron n : hidden ) {
			n.trainingMode = trainingMethod;
		}
		for (Neuron n : output ) {
			n.trainingMode = trainingMethod;
		}
	}

	/**
	 * Training algorithm, see Neuron
	 * @return
	 */
	public int getTrainingAlgorithm() {
		return trainingAlgorithm;
	}

	/**
	 * Applies training algorithm to all neurons in the network
	 * @param trainingAlgorithm
	 */
	public void setTrainingAlgorithm(int trainingAlgorithm) {
		this.trainingAlgorithm = trainingAlgorithm;
		for (Neuron n : hidden ) {
			n.trainingAlgorithm = trainingAlgorithm;
		}
		for (Neuron n : output ) {
			n.trainingAlgorithm = trainingAlgorithm;
		}
	}
	
	public String getStats() {
		return "iterations " + currentIterations + " min error " + minError + " max error " + 
				maxError + " right "  + right + " wrong " + wrong + " success " + ( new Double(right) / new Double(currentIterations) );
				//" avg error " + ( new Double(totalError) / currentSetIterations.doubleValue() ) +
				//" std dev " + Math.sqrt(squareError / currentSetIterations.doubleValue()); 
	}

	@Override
	public int compareTo(Network o) {
		return Double.compare(this.totalError, o.totalError);
	}
	

	
}
