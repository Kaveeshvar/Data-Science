1. What changed under ill-conditioning and poisoning, and why?

Ill-conditioning will cause the predicted to be deviated in random directions. 
Poisoning will cause the error to sky-rocket because the error is squared. Large outliers will weigh more and disturb the data more because of the squaring of mse. Squaring is done so that the positive and negative errors don't cancel out. 

Under Ill-conditioning, the feature matrix remainns near collinear, that is Xtx is almost singular. Geometrically loss surfacce is long, narrow valley with very flat directions. Closed form will generate inflated weight magnitudes while GD converges slowly due to zig-zagging along the poor directions. 

2. When poisoning is introduced, small number of high-magnitude or extreme-label points impact the optimization because of squaring of mse.  In closed form the parameters shift immediately while in GD, the gradient slowly get larger destabilizing training and turning convergence toward a biased optimum.


3. Explain why GD converges slowly when XtX is ill-conditioned. What does feature scaling do geometrically
GD converges more slowly due to zig-zagging along poorly conditioned directions. when Xtx is ill-conditioned, the bowl is very stretched. One side is steep, one side is flat. so gd zig-zags instead of going straight down. Feature scaling makes the bowl rounder. 

4. When would you prefer pseudoinverse/normal equation vs GD/SGD in production? Consider compute, memory, and numerical stability.

pseudoinverse/normal equation : 
    Small dataset
    Few Features
    Enough memory
    Want solution in one shot

GD : 
    Huge dataset
    Many Features
    Memory is limited
    Scalable training


5. Why is MSE regression vulnerable to outliers/poisoning, and what defenses would you implement in a real pipeline?
    MSE regression is vulnerable because of the squaring of errors. Outliers/ Poisoning even in small proportions can impact the score hugely. 
    Defenses : 
        Remove / clip extreme values
        Use robust loss (MAE, Huber)
        Scale features
        Add regularization
        Monitor data for anomalies