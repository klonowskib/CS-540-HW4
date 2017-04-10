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
		double largest = 0;
		ArrayList<Double> output = new ArrayList<Double>();
		int count = 0;
		/* Forward pass */
		for (Node input : inputNodes) {
			//set the input to the corresponding aspect of the instance
			try {
				input.setInput(inst.attributes.get(count));
			}
			catch(IndexOutOfBoundsException e){
				//System.out.println("bad index");
			}
			count++;
		}
		for(Node n : hiddenNodes)
			n.calculateOutput();
		//System.out.println();
		for(Node n : outputNodes)
			n.calculateOutput();
		/* Forward pass */

		int prediction = 0;
		count = 0;
		for(Node n : outputNodes) {
			double out = n.getOutput();
			if(largest <= out) {
				largest = out;
				prediction = count;
			}
			count++;
		}
		//System.out.println(prediction);
		return prediction;
	}


	/**
	 * Train the neural networks with the given parameters
	 *
	 * The parameters are stored as attributes of this class
	 */

	public void train()
	{
		int epoch = 0;
		while (epoch <= maxEpoch) {
			epoch++;
			for (Instance inst : trainingSet) {
				int inst_out = this.calculateOutputForInstance(inst);
				int [] o = new int[inst.classValues.size()];
				for(int tmp : o) {
					tmp = 0;
				}
				o[inst_out] = 1;
				double [] error = new double[outputNodes.size()];
				double [] delta_k = new double[outputNodes.size()];

				int count = 0;
				double [][] delta_jk = new double [hiddenNodes.size()-1][outputNodes.size()];
				double [] delta_j = new double[hiddenNodes.size()];
				for(int k_count =0; k_count < outputNodes.size(); k_count++) {
					error[k_count] = (inst.classValues.get(k_count) - o[k_count]);
					delta_k[k_count] = error[k_count]* outputNodes.get(k_count).g_prime();
					for(int j_count = 0; j_count < hiddenNodes.size()-1; j_count++) {
						Node j = hiddenNodes.get(j_count);
						j.calculateOutput();
						delta_jk[j_count][k_count] = learningRate * j.getOutput() * delta_k[k_count];
						//used to update weights between l=1 and l=0
						delta_j[j_count] += j.g_prime() * outputNodes.get(k_count).getSum() * delta_k[k_count];
					}
				}
				double [][] delta_ij = new double [inputNodes.size()][ hiddenNodes.size()];
				for (int j_count = 0; j_count < hiddenNodes.size()-1; j_count++) {

					try {
						for(int i_count = 0; i_count < inputNodes.size()-1; i_count++) {
							Node i = inputNodes.get(i_count);
							delta_ij[i_count][j_count] = learningRate * i.getOutput() * delta_j[j_count];
							i_count++;
						}
					}
					catch (NullPointerException e) {}
					j_count++;
				}
				try {
					for(int k_count = 0; k_count < delta_jk.length - 1; k_count++ ) {
						for(int j_count =0; j_count < delta_jk[k_count].length -1; j_count++) {
							outputNodes.get(k_count).parents.get(j_count).weight += delta_jk[j_count][k_count];
						}
					}
					for(int j_count = 0; j_count < delta_ij.length - 1; j_count++ ) {
						for(int i_count =0; i_count < delta_ij[j_count].length-1; i_count++) {
							Node j = hiddenNodes.get(j_count);
							j.parents.get(i_count).weight += delta_ij[i_count][j_count];
						}
					}
				}
				catch (NullPointerException e){}

			}
				/* Back propogation */
			//output to hidden
				/*
				int count = 0;
				for(Node j : outputNodes){
					j.setDelta(j.g_prime() * (inst.classValues.get(count) - o[count]));
					for(NodeWeightPair parent : j.parents) {
						Node i = parent.node;
						double value = i.g_prime() * j.getSum() * j.getDelta();
						i.setDelta(value);
					}
					count++;
					j.update_weights(learningRate);
				}
				*/
				/*
				//hidden to input
				for(Node j : hiddenNodes) {
					try {
						for (NodeWeightPair parent : j.parents) {
							Node i = parent.node;
							//System.out.println("hidden to input");
							//double g = 1 / (1 + Math.exp(-i.getSum()));
							//double gp = g * (1 - g);
							//double value = gp * j.getSum() * j.getDelta();
							//i.setDelta(value);
						}
					} catch (NullPointerException e) {}
					j.update_weights(this.learningRate);
				}
				*/

				/* Update weights */
			// weights from hidden to output
				/*for(Node j : outputNodes) {
					try {
						j.update_weights(this.learningRate);
					} catch (NullPointerException e) {
						e.printStackTrace();
					}
				}
				for (Node n : hiddenNodes) {
					try {n.update_weights(this.learningRate);}
					catch (NullPointerException e ){}
				}
				*/

		}
	}
}
