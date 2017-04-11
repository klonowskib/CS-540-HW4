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


	/*public void setDelta (Double value) {
		this.delta = value;
	}
	public double getDelta () {
		return  this.delta;
	}
	*/
	public double delta_w () {return  1.0;}

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

	public void update_weights () {
		try {
			for (NodeWeightPair parent : parents) {
				Node i = parent.node;
				parent.weight += i.getOutput() * delta;
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


	public void setDelta(double delta) {
		this.delta = delta;
	}
	public double getDelta() {
		return this.delta;
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
			this.calc_sum();
			this.outputValue = g(this.sum);
		}
	}

	public double g_prime () {
		return g(this.sum) * (1- g(this.sum));
	}

	public void calc_sum () {
		double tmp_sum = 0;
		for(NodeWeightPair parent: parents)
			tmp_sum += parent.node.getOutput() * parent.weight;
		this.sum = tmp_sum;
	}

	public double g (double x) {
		return 1/(1+Math.exp(-x));
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


