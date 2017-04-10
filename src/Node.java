/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details
 *
 * Do not modify. 
 */


import java.util.ArrayList;

public class Node{
	private int type=0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents=null; //Array List that will contain the parents (including the bias node) with weights if applicable

	private Double inputValue=0.0;
	private Double outputValue=0.0; // Output value of a node: same as input value for an iput node, 1.0 for bias nodes and calculate based on Sigmoid function for hidden and output nodes
	private Double sum=0.0; // sum of wi*xi
	private Double delta = 0.0;


	public void setDelta (Double value) {
		this.delta = value;
	}
	public double getDelta () {
		return  this.delta;
	}


	//Create a node with a specific type
	public Node(int type)
	{
		if(type>4 || type<0)
		{
			System.out.println("Incorrect value for node type");
			System.exit(1);

		}
		else
		{
			this.type=type;
		}

		if (type==2 || type==4)
		{
			parents=new ArrayList<NodeWeightPair>();
		}
	}

	public void update_weights (double alpha) {
		try {
			for (NodeWeightPair parent : parents) {
				Node i = parent.node;
				parent.weight = parent.weight + alpha * i.getOutput() * delta;
			}
		}
		catch (NullPointerException e) {}

	}
	//For an input node sets the input value which will be the value of a particular attribute
	public void setInput(Double inputValue)
	{
		if(type==0)//If input node
		{
			this.inputValue=inputValue;
		}
	}

	/**
	 * Calculate the output of a Sigmoid node.
	 * You can assume that outputs of the parent nodes have already been calculated
	 * You can get this value by using getOutput()
	 */
	public void calculateOutput()
	{
		if(type==2 || type==4)//Not an input or bias node
		{
			int n; //Number of inputs
			double x = 0; //result of the summation to be done below
			double g = 0; //placeholder for output value

			//Calculate the x value for the sigmoid
			for (NodeWeightPair input : this.parents) {
				x += (input.weight * input.node.getOutput());
				//System.out.println("weight " + input.weight + " input " + input.node.getOutput()+ " sum " + x + " type " + type);
			}
			this.sum = x;
			//Sigmoid activation function
			g = 1/(1+ Math.exp(-this.sum));
			//System.out.println("result " + g + " sum: " + sum + " type " + type);
			this.outputValue = g;
		}
	}

	public double g_prime () {
		return this.getOutput() * (1- this.getOutput());
	}

	public double getSum() {
		return sum;
	}

	//Gets the output value
	public double getOutput()
	{

		if(type==0)//Input node
		{
			return inputValue;
		}
		else if(type==1 || type==3)//Bias node
		{
			return 1.00;
		}
		else
		{
			return outputValue;
		}

	}
}


