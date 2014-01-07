import java.io.IOException;
import java.util.PriorityQueue;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.greatnowhere.neural.NetworkTrainer;
import com.greatnowhere.neural.Neuron;



@RunWith(JUnit4.class)
public class XORTest {

	public static final int NUM_INPUT_NEURONS = 2;
	public static final int NUM_HIDDEN_NEURONS = 2;
	public static final int NUM_OUTPUT_NEURONS = 1;
	public static final int NUM_ITERATIONS = 3000;
	public static final int NUM_SPECIMENS = 10;
	public static final int NUM_SHEEP = 100;
	public static final int NUM_GENERATIONS = 10;
	public static final double LEARNING_RATE = 0.5D;
	public static final double MOMENTUM = 0.2D;
	
	
	@Test
	@Ignore
	public void test() throws InterruptedException, IOException {
		
		NetworkTrainer t_best = null;	
		
		// XOR test
		do {
			PriorityQueue<NetworkTrainer> trainers_temp = new PriorityQueue<>();	
			for (int i=0; i<10; i++) {
				NetworkTrainer t = getXORTrainer();
				t.train(null);
				// System.out.println("success rate " + t.successRate);
				trainers_temp.add(t);
			}
			NetworkTrainer t1 = trainers_temp.poll();
			if ( t1.n.successRate > 0.8D ) break;
			if ( t_best == null ) 
				t_best = t1;
			t_best.merge(t1);
			t_best.train(null);
			System.out.println("success rate " + t_best.n.successRate + " error " + t_best.n.totalError);
			System.out.println(t_best.n.toString());
			
		} while ( t_best.n.successRate < 0.9D );

		
	}
	
	@Test
	public void stochasticBackpropTrain() throws InterruptedException, IOException {
		
		NetworkTrainer t = getXORTrainer();	
		
		// XOR test
		t.n.setTrainingAlgorithm(Neuron.TRAINING_ALG_BACKPROP);
		t.n.setTrainingMethod(Neuron.TRAINING_MODE_STOCHASTIC);

		//Assert.assertTrue("Stochastic backprop training not converging", t.trainGenetic(null, NUM_ITERATIONS, 1, NUM_SPECIMENS, NUM_GENERATIONS, true));
		Assert.assertTrue("Stochastic backprop training not converging", 
				t.trainHerd(null, NUM_ITERATIONS, 1, NUM_SHEEP, true));
		
	}
	
	@Test
	public void batchBackpropTrain() throws InterruptedException, IOException {
		
		NetworkTrainer t = getXORTrainer();	
		
		// XOR test
		t.n.setTrainingAlgorithm(Neuron.TRAINING_ALG_BACKPROP);
		t.n.setTrainingMethod(Neuron.TRAINING_MODE_BATCH);

		//Assert.assertTrue("Batch backprop training not converging", t.trainGenetic(null, NUM_ITERATIONS, 1, NUM_SPECIMENS, NUM_GENERATIONS, true));
		Assert.assertTrue("Stochastic backprop training not converging", 
				t.trainHerd(null, NUM_ITERATIONS, 1, NUM_SHEEP, true));
		
	}
	
	@Test
	public void batchRpropTrain() throws InterruptedException, IOException {
		
		NetworkTrainer t = getXORTrainer();	
		
		// XOR test
		t.n.setTrainingAlgorithm(Neuron.TRAINING_ALG_RPROP);
		t.n.setTrainingMethod(Neuron.TRAINING_MODE_BATCH);

		//Assert.assertTrue("Batch rprop training not converging", t.trainGenetic(null, NUM_ITERATIONS, 1, NUM_SPECIMENS, NUM_GENERATIONS, true));
		Assert.assertTrue("Stochastic backprop training not converging", 
				t.trainHerd(null, NUM_ITERATIONS, 1, NUM_SHEEP, true));
		
	}
	
	@Test
	public void stochasticRpropTrain() throws InterruptedException, IOException {
		
		NetworkTrainer t = getXORTrainer();	
		
		// XOR test
		t.n.setTrainingAlgorithm(Neuron.TRAINING_ALG_RPROP);
		t.n.setTrainingMethod(Neuron.TRAINING_MODE_STOCHASTIC);

		//Assert.assertTrue("Stochastic rprop training not converging", t.trainGenetic(null, NUM_ITERATIONS, 1, NUM_SPECIMENS, NUM_GENERATIONS, true));
		Assert.assertTrue("Stochastic backprop training not converging", 
				t.trainHerd(null, NUM_ITERATIONS, 1, NUM_SHEEP, true));
		
	}
	
	NetworkTrainer getXORTrainer() {
		NetworkTrainer t = new NetworkTrainer();
		
		t.init(NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM);
		t.addTrainingSet(new double[] {0D, 0D}, 0D, "0 xor 0 = 0");
		t.addTrainingSet(new double[] {1D, 0D}, 1D, "1 xor 0 = 1");
		t.addTrainingSet(new double[] {0D, 1D}, 1D, "0 xor 1 = 1");
		t.addTrainingSet(new double[] {1D, 1D}, 0D, "1 xor 1 = 0");
		
		return t;
	}
	
	NetworkTrainer getANDTrainer() {
		NetworkTrainer t = new NetworkTrainer();
		
		t.init(NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM);
		t.addTrainingSet(new double[] {0D, 0D}, 0D, "0 and 0 = 0");
		t.addTrainingSet(new double[] {1D, 0D}, 0D, "1 and 0 = 0");
		t.addTrainingSet(new double[] {0D, 1D}, 0D, "0 and 1 = 0");
		t.addTrainingSet(new double[] {1D, 1D}, 1D, "1 and 1 = 1");
		
		return t;
	}

	NetworkTrainer getORTrainer() {
		NetworkTrainer t = new NetworkTrainer();
		
		t.init(NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM);
		t.addTrainingSet(new double[] {0D, 0D}, 0D, "0 or 0 = 0");
		t.addTrainingSet(new double[] {1D, 0D}, 1D, "1 or 0 = 1");
		t.addTrainingSet(new double[] {0D, 1D}, 1D, "0 or 1 = 1");
		t.addTrainingSet(new double[] {1D, 1D}, 1D, "1 or 1 = 1");
		
		return t;
	}

}
