
# Multi-Objective Optimisation of Spiking Neural Networks

## Abstract
Spiking neural networks (SNNs) communicate through the all-or-none spiking activity of neurons. However, fitting the large number of SNN model parameters to observed neural activity patterns, e.g. in biological experiments, remains a challenge. Previous work using genetic algorithm (GA) optimisation on a specific efficient SNN model, using the Izhikevich neuronal model, was limited to a single parameter and objective. This work applied a version of GA, called non-dominated sorting GA (NSGA-III), to demonstrate the feasibility of performing multi-objective optimisation on the same SNN. We focus on searching for network connectivity parameters to achieve target firing rates of excitatory and inhibitory neuronal types, including across different network connectivity sparsity. We showed that NSGA-III could readily optimise for various firing rates. Notably, when the excitatory neural firing rates were higher than or equal to that of inhibitory neurons, the errors were small. Moreover, when connectivity sparsity was considered as a parameter to be optimised, the optimal solutions required sparse network connectivity. We also found that for excitatory neural firing rates lower than that of inhibitory neurons, the errors were generally larger. Overall, we have successfully demonstrated the feasibility of implementing multi-objective GA optimisation on network parameters of recurrent and sparse SNN.

![SNN Firing](snn_firing.png)

