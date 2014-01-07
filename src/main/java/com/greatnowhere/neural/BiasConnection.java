package com.greatnowhere.neural;

/**
 * Defines connection to a bias neuron
 * @author pzeltins
 *
 */
public class BiasConnection extends Connection {
	
	BiasConnection() {
		super();
		source = new BiasInput();
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	class BiasInput extends InputValue {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		public double getOutput() {
			return 1D;
		}
		
	}

}
