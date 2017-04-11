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
		ArrayList<Double> output = new ArrayList<Double>();
		int count = 0;
		/* Forward pass */
		for (Node input : inputNodes) {
			//set the input to the corresponding aspect of the instance
			try {
				input.setInput(inst.attributes.get(count));
			}
			catch(IndexOutOfBoundsException e){}
			count++;
		}
		for(Node n : hiddenNodes)
			n.calculateOutput();
		//System.out.println();
		for(Node n : outputNodes)
			n.calculateOutput();
		/* Forward pass */

		int prediction = 0;
		double largest = -1;
		count = 0;
		for(Node n : outputNodes) {
			if(largest <= n.getOutput()) {
				largest = n.getOutput();
				prediction = count;
			}
			count++;
		}
		//	System.out.println(prediction);
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

				/* Back propogation */
				//output to hidden
				/*int count = 0;
				for(Node k : outputNodes){
					for(NodeWeightPair parent : k.parents) {
						Node j = parent.node;
						k.setDelta(learningRate * j.getOutput() * (inst.classValues.get(count) - o[count])*o[count]*(1-o[count]));
						double value = j.g_prime() * k.getSum() * k.getDelta();
						j.setDelta(value);
					}
					count++;
					//	j.update_weights();
				}

				//hidden to input
				for(Node j : hiddenNodes) {
					try {
						for (NodeWeightPair parent : j.parents) {
							Node i = parent.node;

							//System.out.println("hidden to input");
						}
					} catch (NullPointerException e) {}
					//j.update_weights();
				}

				/* Update weights */
				// weights from hidden to output
				double a = learningRate;
				double jk_sum = 0;
				int count = 0;
				for(Node k : outputNodes) {
					k.calculateOutput();
					double k_delta = (inst.classValues.get(count) - k.getOutput()) * k.getOutput() * (1 - k.getOutput());
					k.setDelta(k_delta);
					for (NodeWeightPair hidden : k.parents) {
						Node j = hidden.node;
						jk_sum += hidden.weight * k.getDelta();
						j.setDelta(j.g_prime() * jk_sum);
						hidden.weight += a * j.getOutput() * k.getDelta();
					}
					count ++;
				}
				count = 0;
				for (Node j : hiddenNodes) {
					//hidden.weight += learningRate * j.getOutput() * (inst.classValues.get(count) - o[count]) * k.g_prime();
					try {
						for(NodeWeightPair i : j.parents) {
							double iout = i.node.getOutput();
							i.weight += a *  iout * j.getDelta();
						}
					}
					catch (NullPointerException e ){}
				}
				count ++;
			}
		}
	}
}
