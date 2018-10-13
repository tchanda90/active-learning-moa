
package moa.classifiers.active;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.BetaDistribution;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.active.ALClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import weka.core.Utils;

/**
* Active Learning with Self-Labeling. 
* 
* Reference: Lukasz Korycki1and, Bartosz Krawczyk, Combining Active Learning 
* and Self-Labeling for Data Stream Mining
*
*/

public class ALSelfLabeling extends AbstractClassifier implements ALClassifier{
	
	private static final long serialVersionUID = 1L;

	@Override
    public String getPurposeString() {
        return "Active learning algorithm with self labeling";
    }
	
	/**
	 * Base classifier option
	 */
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Base classifier to use", Classifier.class, "trees.HoeffdingAdaptiveTree");
	
	/**
	 * Type of threshold randomization.
	 * Beta, normal, or uniform distribution
	 */
	public MultiChoiceOption thresholdRandomizationOption = new MultiChoiceOption(
            "thresholdRandomization", 'r', "Threshold randomization strategy.",
            new String[]{
                    "betaDistribution",
                    "normalDistribution",
                    "uniformDistribution"},
            new String[]{
                    "Beta distribution with alpha 2 and beta 2",
                    "Normal distribution with mean 1 and std 1",
                    "Uniform distribution for randomization of the threshold"}, 
            0);
	
	/**
	 * Number of initial instances (will be deducted from budget).
	 */
	public IntOption numInitInstancesOption = new IntOption("numInitInstances", 'n',
			"Number of initial instances to train on", 
			0, 0, Integer.MAX_VALUE);

	/**
	 * Budget as a factor of number of instances processed
	 */
	public FloatOption budgetOption = new FloatOption("budget", 'b', 
			"Budget as a factor of number of instances",
            0.1, 0.0, 1.0);
	
	/**
	 * Threshold value.
	 */
	public FloatOption thresholdOption = new FloatOption("threshold",
            't', "Initial threshold value for querying labels",
            1, 0, 1);
    
	/**
	 * Confidence level value between 0 and 1. If the maximum posterior is greater or equal 
	 * to the confidence, the instance can be used for self labeling.
	 */
    public FloatOption confidenceOption = new FloatOption("confidence",
            'c', "Confidence threshold to decide if instance can be self-labeled",
            0.9, 0, 1.1);
    
    /**
     * Step option
     */
    public FloatOption stepOption = new FloatOption("step",
            's', "Threshold adjustment step",
            0.01, 0, 1);

    protected Classifier classifier;  
    protected int numInstances; 
    protected int numInitInstances;
    protected int numSelfLabeled;
    protected int lastLabelAcq;
    protected int labelsAcquired;
    protected double currentSpent;
    protected double argMax;
    
    protected double budget;
    protected double threshold;
    protected double thresholdR;
    protected double randomMultiplier;
    protected double confidence;
    protected double step;
    
    protected BetaDistribution betaDistribution;
    protected Random distribution;
    
    /*
    int numGreater;
    int numLesser;
    */

    @Override
    public void resetLearningImpl() {
    	
    	this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption));
    	this.classifier.resetLearning();
    	
    	this.lastLabelAcq = 0;
    	
    	this.numInstances = 0;
    	this.labelsAcquired = 0;
    	this.numInitInstances = this.numInitInstancesOption.getValue();
    	this.budget = this.budgetOption.getValue();
    	this.threshold = this.thresholdOption.getValue();
    	this.thresholdR = 0;
    	this.randomMultiplier = 0;
    	this.confidence = this.confidenceOption.getValue();
    	this.step = this.stepOption.getValue();
    	this.currentSpent = 0;
    	this.argMax = 0;
    	this.numSelfLabeled = 0;
    	
    	this.betaDistribution = new BetaDistribution(2, 2);
    	this.distribution = new Random();		
    }


    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	
      	// counter for number of instances
      	++this.numInstances;
      	
      	// train initial examples and increment the number of labels acquired
      	if (this.numInstances <= this.numInitInstances) {
      		++this.labelsAcquired;
      		classifier.trainOnInstance(inst);
    		return;
      	}
      	
      	// calculates the current spent budget as a factor of the number of instances
    	this.currentSpent = (this.labelsAcquired / ((double) this.numInstances));
    	
    	// if budget available
    	if (this.currentSpent < this.budget) {
    		
    		// get argmax of the posteriors
    		this.argMax = getArgMax(this.classifier.getVotesForInstance(inst));
    		
    		// check if instance should be queried
	    	if (queryLabel(this.argMax, inst) == true) {
	    		this.lastLabelAcq += 1;
	    		this.classifier.trainOnInstance(inst);
	    		++this.labelsAcquired ;
	    		return;
	    	}
    	}
    	// check if self labeling is possible
    	if (canSelfLabel(this.argMax, inst) == true) {
    		// set the classifier's prediction as the class label and train
    		int trueLabel = Utils.maxIndex(getVotesForInstance(inst));
    		inst.setClassValue(trueLabel);
    		this.classifier.trainOnInstance(inst);
    		++this.numSelfLabeled;
    	}
    }

    
    /**
     * Checks if the maximum posterior is below the threshold. If yes, returns true, which means
	 * that the true label should be queried. Else returns false. The uncertainty region is 
	 * also decreased if true and increased if false
     */
	public boolean queryLabel(double argMax, Instance inst) {
	
		switch (this.thresholdRandomizationOption.getChosenIndex()) {		
			
			case 0: // beta
				this.randomMultiplier = betaDistribution.sample() + 0.5;
				break;			
			
			case 1: // normal
				// sometimes may generate values <= 0, so keep generating till values > 0
				do {
					this.randomMultiplier = distribution.nextGaussian() * 1 + 1;
				} while (this.randomMultiplier <= 0);
				break;	
			
			case 2: // uniform
				do {
					this.randomMultiplier = distribution.nextDouble();
				} while (this.randomMultiplier <= 0);
				break;
		}
		
		// threshold randomization
		this.thresholdR = this.threshold * this.randomMultiplier;
		
		if (argMax < thresholdR) {
			this.threshold = this.threshold * (1 - this.step);
			return true;
		} else {
			this.threshold = this.threshold * (1 + this.step);
			return false;
		}		
	}
    
    
	/**
	 * If the max posterior is greater than or equal to the confidence level, the instance
	 * can be self labeled and used for training
	 */
    private boolean canSelfLabel(double argMax, Instance inst) {
		if (argMax >= this.confidence) {
			return true;
		} else {
			return false;
		}
	}

    
    /**
     * Takes as input an array of probabilities.
     * Normalizes the probabilities to sum to 1 and returns the max probability 
     */
    private double getArgMax(double[] incomingPrediction) {
    	double argMax;
        if (incomingPrediction.length > 1) {
            DoubleVector vote = new DoubleVector(incomingPrediction);
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
            }
            incomingPrediction = vote.getArrayRef();
            argMax = (incomingPrediction[Utils.maxIndex(incomingPrediction)]);
        } else {
            argMax = 0;
        }
        return argMax;
    }
    
    
	@Override
	public double[] getVotesForInstance(Instance inst) {
		return this.classifier.getVotesForInstance(inst);
	}
    
        

	@Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
    	List<Measurement> measurementList = new LinkedList<Measurement>();
        measurementList.add(new Measurement("labelAcquired", this.labelsAcquired));
        measurementList.add(new Measurement("numSelfLabeled", this.numSelfLabeled));
        measurementList.add(new Measurement("threshold", this.threshold));
        measurementList.add(new Measurement("thresholdR", this.thresholdR));
        measurementList.add(new Measurement("randomMultiplier", this.randomMultiplier));
        measurementList.add(new Measurement("currentSpent", this.currentSpent));
        measurementList.add(new Measurement("argMax", this.argMax));
        Measurement[] modelMeasurements = ((AbstractClassifier) this.classifier).getModelMeasurements();
        if (modelMeasurements != null) {
            for (Measurement measurement : modelMeasurements) {
                measurementList.add(measurement);
            }
        }
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
	public int getLastLabelAcqReport() {
    	int num = this.lastLabelAcq;
		this.lastLabelAcq = 0;
		return num; 
	}
	
	@Override
	public void setModelContext(InstancesHeader ih) {
		super.setModelContext(ih);
		classifier.setModelContext(ih);
	}
}
