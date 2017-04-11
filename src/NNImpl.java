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
				calculateOutputForInstance(inst);
				// O = out put of neural net from
				ArrayList<Double> O = new ArrayList<>(5);
				ArrayList<Double> error = new ArrayList<>(5);
				Double[][] w_ij = new Double[inputNodes.size()][hiddenNodes.size()];
				Double[][] w_jk = new Double[hiddenNodes.size()][5];
				for (int idx = 0; idx < outputNodes.size(); idx++) {
					O.add(idx, outputNodes.get(idx).getOutput());
				}

				// T = desired output, i.e. Target or Teachers output
				// calculate error (Tk - Ok) at each output unit k
				for (int idx = 0; idx < 5; idx++) {
					error.add(idx, inst.classValues.get(idx) - O.get(idx));
				}
				// for each hidden unit j and output unit k compute
				for (int j = 0; j < hiddenNodes.size(); j++) {
					for (int k = 0; k < outputNodes.size(); k++) {
						// delta wjk = alpha * aj * (Tk - Ok) g'(in_k)
						w_jk[j][k] = learningRate * hiddenNodes.get(j).getSum() * error.get(k) * g_prime(outputNodes.get(k).getSum());
					}
				}

				// for each input unit i and hidden unit j compute
				for (int i = 0; i < inputNodes.size()-1; i++) {
					for (int j = 0; j < hiddenNodes.size()-1; j++) {
						// delta wij = alpha * ai * g'(in_j)*sum[wjk *(Tk-Ok) * g'(in_k)]
						double sumK = 0.0;
						for (int k = 0; k < outputNodes.size(); k++) {
							sumK += outputNodes.get(k).parents.get(j).weight * error.get(k) * g_prime(outputNodes.get(k).getSum());
						}
						w_ij[i][j] = learningRate * inst.attributes.get(i) * g_prime(hiddenNodes.get(j).getSum()) * sumK;
					}
				}
				// for all p, q in network wpq = wpq + delta wpq
				// update in->hidden
				for (int q = 0; q < hiddenNodes.size()-1; q++) {
					for (int p = 0; p < inputNodes.size()-1; p++) {
						// System.out.println(hidW[p][q]);
						hiddenNodes.get(q).parents.get(p).weight += w_ij[p][q];
					}
				}

				// update hidden->out
				for (int q = 0; q < outputNodes.size(); q++) {
					for (int p = 0; p < hiddenNodes.size(); p++) {
						outputNodes.get(q).parents.get(p).weight += w_jk[p][q];
					}
				}
			}
		}
	}

	public double g (double x) {
		return 1/(1+Math.exp(-x));
	}
	public double g_prime(double x) {
		return g(x)*(1-g(x));
	}
}
