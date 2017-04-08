/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details
 * 
 * Do not modify. 
 */


import java.util.*;

public class Node{
	private int type=0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents=null; //Array List that will contain the parents (including the bias node) with weights if applicable
		 
	private Double inputValue=0.0;
	private Double outputValue=0.0; // Output value of a node: same as input value for an iput node, 1.0 for bias nodes and calculate based on Sigmoid function for hidden and output nodes
	private Double sum=0.0; // sum of wi*xi
	private Double delta_j = 0.0;
	private Double delta_i = 0.0;

	public void setDelta_j (Double value) {
		this.delta_j = value;
	}
	public double getDelta_j () {
		return  this.delta_j;
	}

	public void setDelta_i (Double value) {
		this.delta_i = value;
	}
	public double getDelta_i () {
		return  this.delta_i;
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
			// TODO: add code here
			int n; //Number of inputs
			double x = 0; //result of the summation to be done below
			double g = 0; //placeholder for output value

			//Calculate the x value for the sigmoid
			for (NodeWeightPair input : this.parents)
				x += input.weight * input.node.outputValue;
			//Sigmoid activation function
			this.sum = x;
			g = 1/(1+ Math.exp(-x));
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


