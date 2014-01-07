package com.greatnowhere.neural;

import java.io.Serializable;
import java.util.Random;

/**
 * defines a generic weighted connection between 2 neurons 
 * @author pzeltins
 *
 */
public class Connection implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public static final double RPROP_INITIAL_UPDATE_VALUE = 0.1D;
	
	public IOutput source;
	public Neuron dest;
	public double destWeight = 2D * (r.nextDouble() - 0.5D);
	/**
	 * Last change of the weight. Used for momentum calculation
	 */
	public double deltaWeight = 0D;
	/**
	 * Total accumulated weight delta in batch learning cycle 
	 */
	public double totalDeltaWeight = 0D; 
	/**
	 * Total accumulated weight delta in previous batch learning cycle. Used in RPROP
	 */
	public double totalDeltaWeightPrev = 0D; // for RPROP learning
	/**
	 * update value for RPROP algorithm
	 */
	public double updateValue = RPROP_INITIAL_UPDATE_VALUE; 
	/**
	 * previous update value for RPROP algorithm
	 */
	public double prevDeltaWeight = 0D; 
	/**
	 * RPROP min update value
	 */
	public final static double MIN_RPROP_UPDATE_VALUE = 1E-6D;
	/**
	 * RPROP max update value
	 */
	public final static double MAX_RPROP_UPDATE_VALUE = 50;
	
	static Random r = new Random();
	
	/**
	 * Merges this connection with target connection
	 * @param c
	 */
	public void merge(Connection c) {
		destWeight = ( destWeight + c.destWeight ) / 2D;
		deltaWeight = ( deltaWeight + c.deltaWeight ) / 2D;
		totalDeltaWeight = ( totalDeltaWeight + c.totalDeltaWeight ) / 2D;
	}
	
	/**
	 * Pretty-print this connection's details
	 */
	public String toString() {
		String s = " ";
		if ( source instanceof InputValue ) s+= "i";
		if ( this instanceof BiasConnection ) s+= "b";
		s += String.format("%.2f", destWeight);
		return s;
	}
	
	public boolean equals(Object o) {
		return ( o != null && o instanceof Connection && 
				((Connection) o).source == this.source &&
				((Connection) o).dest == this.dest);
	}
}
