/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the input layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes
	
	public ArrayList<Instance> trainingSet=null;//the training set
	
	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. 
 	*/
	
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}
		
		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);
			
		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}
	
	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2, 0.1, 0.1], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.1, 0.5, 0.2], it should return 3. 
	 * The parameter is a single instance. 
	 */
	
	public int calculateOutputForInstance(Instance inst)
	{
		// TODO: add code here
		//Calculates the output given an example (inst)
		int count = 0;
		int output = -1;
		double largest = -1000;
		for(Node out : this.outputNodes) {
			if (out.getOutput() >= largest) {
				switch (count) {
					case 0:
						output = 1;
					case 1:
						output = 4;
					case 2:
						output = 7;
					case 3:
						output = 8;
					case 4:
						output = 9;
					default:
						return -1;
				}
			}
			count++;
		}
		System.out.println(output);
		if (output == -1)
			System.out.println("This should never be reached");
		return output;
	}

	public int calculateClass (Instance inst) {
		int count = 0;
		int output = -1;
		double largest = 0;
		for(double classValue : inst.classValues) {
			if (classValue == 1) {
				switch (count) {
					case 0:
						output = 1;
					case 1:
						output = 4;
					case 2:
						output = 7;
					case 3:
						output = 8;
					case 4:
						output = 9;
					default:
						return -1;
				}
			}
			count++;
		}
		System.out.println(output);
		if (output == -1)
			System.out.println("Invalid output value for this instance");
		return output;
	}
	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */
	
	public void train()
	{
		// TODO: add code here
		//Given a training set, fixed learning rate, and number of epochs train the neural network
		//Adjust weights

		//For each training point
		for(Instance inst : this.trainingSet) {
			int count = 0;
			for(Node input : this.inputNodes)
				input.setInput(inst.attributes.get(count));
			for(Node hidden : hiddenNodes) {

			}
		}
	}
}
