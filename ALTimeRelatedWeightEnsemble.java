package moa.classifiers.active;

import java.util.Arrays;
import java.util.List;
import java.util.LinkedList;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.active.ALClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import scala.Console;
import weka.core.Utils;

public class ALTimeRelatedWeightEnsemble extends AbstractClassifier implements ALClassifier
{
	//private static final long serialVersionUID = 1L;
	
	@Override
    public String getPurposeString() {
        return "Active learning classifier based on ensemble time-related weight framework by Shan, Chu, Liu, Dai and Liu";
    }
	
	//----------------------GUI OPTIONS--------------------------------
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train", Classifier.class, "bayes.NaiveBayes");
		
	public FloatOption budgetOption = new FloatOption("budget",
            'b', "Budget to use",
            0.1, 0.0, 1.0);
	
	public FloatOption randomThresholdOption = new FloatOption("randomThreshold",
            't', "Threshold to sample randomly",
            0.25, 0.0, 1.0);
	
	public IntOption maxBaseClassifiersOption = new IntOption("numberOfMaxBaseClassifiers", 'c',
			"Number of maximum base classifiers", 10);
	
	public IntOption chunkSizeOption = new IntOption("chunkSize", 's',
			"Size of chunk per classifier", 500);
	
	public FloatOption stableWeightOption = new FloatOption("stableClassifierWeight", 'w',
			"Weight of stable classifier against base classifiers (sum base weights == 1)", 1.0);
			
	
	//--------------------VARS---------------------
	//Remember: update resetLearningImpl() accordingly!
	public double certainty = 0; //for uncertainty strategy
	public int strategyApplied = 0; //-1 insufficient budget, 0 none, 1 uncertainty, 2 random
	public int noOfInstances = 0; //number of instances
	public int noOfLabeled = 0; //number of labeled instances
	public int noOfChunks = 0; //number of chunks created == n
	public int noOfClassifiers = 1; //number of classifiers == k + 1
	public int lastLabelAcq = 0; //how many labels have been labeled since getLastLabelAcqReport was called
	public int[] firstLabeledInstanceOfChunk;
	public Classifier[] classifiers; // == C_0, C_n-k+1, C_n-k+2, ... , C_n
	public double[] weights; // == w_0, w_n-k+1, w_n-k+2, ... , w_n 
	
	@Override
	public boolean isRandomizable() {
		return true; //true because we must use a RNG
	}
	//higher than if last instance was labeled
	@Override
	public int getLastLabelAcqReport() {
		int help = this.lastLabelAcq;
		this.lastLabelAcq = 0;
		return help; 
	}
	//returns judgement of ensemble of instance being any of the classes
	@Override
	public double[] getVotesForInstance(Instance inst) {
		DoubleVector final_votes = new DoubleVector();
		
		for(int i = 0; i < this.noOfClassifiers; i++) {
			DoubleVector vote = new DoubleVector(this.classifiers[i].getVotesForInstance(inst));
			vote.scaleValues(this.weights[i]);
			final_votes.addValues(vote);
		}
		
		final_votes.normalize();
		
		return final_votes.getArrayRef();
	}
	//resets class for new run
	@Override
	public void resetLearningImpl() {
		this.classifiers = new Classifier[this.maxBaseClassifiersOption.getValue() + 1];
		this.weights = new double[this.maxBaseClassifiersOption.getValue() + 1];
		this.classifiers[0] = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
		this.classifiers[0].resetLearning();
		this.weights[0] = this.stableWeightOption.getValue();
		this.noOfInstances = 0;
		this.noOfLabeled = 0;
		this.noOfChunks = 0;
		this.noOfClassifiers = 1;		
		this.lastLabelAcq = 0;
		this.firstLabeledInstanceOfChunk = new int[this.maxBaseClassifiersOption.getValue() + 1];
	}
	//called whenever a new instance comes in
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		this.noOfInstances++;
		this.strategyApplied = -1;
		this.certainty = -1;
		
		if (this.noOfInstances % this.chunkSizeOption.getValue() == 1) {
			//new chunk
			this.noOfChunks++;
			if (this.noOfClassifiers > this.maxBaseClassifiersOption.getValue()) {
				//shift all base classifiers one left; erase oldest
				for(int i = 1; i < this.noOfClassifiers - 1; i++) {
					this.classifiers[i] = this.classifiers[i+1];
					this.weights[i] = this.weights[i+1];
					this.firstLabeledInstanceOfChunk[i] = this.firstLabeledInstanceOfChunk[i+1];
				}
				this.noOfClassifiers--;
			}
			
			addClassifier();
			
			labelInstance(inst);
			this.firstLabeledInstanceOfChunk[this.noOfClassifiers - 1] = this.noOfLabeled;
			
			this.weights[this.noOfClassifiers - 1] = 1.0 / (this.noOfClassifiers - 1);
			
			//decay base weights according to their age
			for(int i = 1; i < this.noOfClassifiers; i++)
				this.weights[i] *= 1 - Math.log(1 + (this.noOfClassifiers - 1 - i) / (double)this.noOfClassifiers);
			normalizeWeights();
		} else {
			if (withinBudget())
				if (!uncertaintyStrategy(inst)) {
					randomStrategy(inst);
				}
		}
	}

	//adds a classifier based on chosen baseLearner
	private void addClassifier() {
		this.classifiers[this.noOfClassifiers] = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
		this.classifiers[this.noOfClassifiers].resetLearning();
		this.weights[this.noOfClassifiers] = 1.0 / (this.noOfClassifiers + 1);
		this.noOfClassifiers++;
	}
	
	private void labelInstance(Instance inst) {
		this.classifiers[0].trainOnInstance(inst);
		this.classifiers[this.noOfClassifiers - 1].trainOnInstance(inst);
		this.noOfLabeled++;
		this.lastLabelAcq += 1;
	}
	
	private void normalizeWeights() {		
		//with stable -----------------------------------
		//this.weights[0] = this.baseWeightOption.getValue();
		//DoubleVector vec_weights = new DoubleVector(this.weights);
		//vec_weights.normalize();
		//this.weights = vec_weights.getArrayRef();
		
		//without stable -------------------------------
		DoubleVector vec_weights = new DoubleVector(Arrays.copyOfRange(this.weights, 1, this.noOfClassifiers));
		vec_weights.normalize();
		for(int i = 0; i < vec_weights.numValues(); i++)
			this.weights[i+1] = vec_weights.getValue(i);
	}
	
	//true if instance can be labeled budget wise
	private boolean withinBudget() {
		return (budgetOption.getValue() > (this.noOfLabeled + 1) / (double)this.noOfInstances);
	}
	
	//true if strategy applied
	private boolean uncertaintyStrategy(Instance inst) {
		double[] posteriors = this.getVotesForInstance(inst);
		
		//failsave for posterior == 1
		if (posteriors.length == 1) {
			labelInstance(inst);
			this.certainty = 1;
			return false;
		}
		
		//normalize via DoubleVector
		DoubleVector vote = new DoubleVector(posteriors);
        vote.normalize();
        posteriors = vote.getArrayRef();
        
        //sort descending with sort and reverse because java
		Arrays.sort(posteriors);
		//reverse array
		for(int i = 0; i < posteriors.length / 2; i++)
		{
		    double temp = posteriors[i];
		    posteriors[i] = posteriors[posteriors.length - i - 1];
		    posteriors[posteriors.length - i - 1] = temp;
		}
		
		this.certainty = posteriors[0] - posteriors[1];
		if (this.certainty < 0.3 / posteriors.length) {
			//uncertainty strategy applied
			labelInstance(inst);
			this.strategyApplied = 1;
			return true;
		}
		return false;
	}
	
	private void randomStrategy(Instance inst) {
		this.strategyApplied = 0;
		if (this.classifierRandom.nextDouble() < this.randomThresholdOption.getValue()) {
			//random strategy applied
			labelInstance(inst);
			this.strategyApplied = 2;
			//adjust weights accordingly
			for(int i = 1; i < this.noOfClassifiers; i++)
				updateWeight(i, inst.classValue() == Utils.maxIndex(this.classifiers[i].getVotesForInstance(inst)));
			normalizeWeights();
		}
	}
	
	private void updateWeight(int index, boolean increase) {
		if (increase)
			this.weights[index] *= 1 + Math.log(1 + 1.0 / (this.noOfClassifiers - 1));
		else
			this.weights[index] *= 1 - Math.log(1 + 1.0 / (this.noOfClassifiers - 1));
	}
	
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		List<Measurement> measurementList = new LinkedList<Measurement>();
		
		//measurementList.add(new Measurement("budgetUsed", (1.0 * this.noOfLabeled) / this.noOfInstances));
		measurementList.add(new Measurement("chunks", this.noOfChunks));
		measurementList.add(new Measurement("labeledInstancesOfLastCompletedChunk", 
				this.firstLabeledInstanceOfChunk[this.noOfClassifiers - 1] - this.firstLabeledInstanceOfChunk[this.noOfClassifiers - 2]));
		measurementList.add(new Measurement("strategyApplied", this.strategyApplied));
		measurementList.add(new Measurement("certainty", this.certainty));
		measurementList.add(new Measurement("noOfClassifiers", this.noOfClassifiers));
		for(int i = 0; i < this.maxBaseClassifiersOption.getValue() + 1; i++)
			measurementList.add(new Measurement("weight" + i, this.weights[i]));
		return measurementList.toArray(new Measurement[measurementList.size()]);
	}
	@Override
	public void getModelDescription(StringBuilder out, int indent) {

	}
}