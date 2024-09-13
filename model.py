#!/usr/bin/env python
#========================================================================
# StochasticMLP Model Construction
#========================================================================

import math
import sys
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.math as tfm
from tensorflow.keras import Model, layers, optimizers

def normal_logpdf(x, mu, std):
    
    return -0.5 * ((x - mu) / std)**2 - tfm.log(std) - 0.5 * tfm.log(2 * math.pi)

def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
    '''
    computes the log normalizing constant of a continuous Bernoulli distribution in a numerically stable way.
    returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1) and a Taylor approximation in
    [l_lim, u_lim].
    cut_y below might appear useless, but it is important to not evaluate log_norm near 0.5 as tf.where evaluates
    both options, regardless of the value of the condition.
    '''
    
    cut_lam = tf.where(tfm.logical_or(tfm.less(lam, l_lim), tfm.greater(lam, u_lim)), lam, l_lim * tf.ones_like(lam))
    log_norm = tfm.log(tfm.abs(2.0 * tfm.atanh(1 - 2.0 * cut_lam))) - tfm.log(tfm.abs(1 - 2.0 * cut_lam))
    taylor = tfm.log(2.0) + 4.0 / 3.0 * tfm.pow(lam - 0.5, 2) + 104.0 / 45.0 * tfm.pow(lam - 0.5, 4)
    return tf.where(tfm.logical_or(tfm.less(lam, l_lim), tfm.greater(lam, u_lim)), log_norm, taylor)

class StochasticMLP(Model):
    
    def __init__(self, hidden_layer_sizes=[32], n_outputs=10, lr=1e-4, is_gibbs=False, L=-1):
        '''
        Initialize the network.
        '''
        super(StochasticMLP, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        if isinstance(self.hidden_layer_sizes, (int, float)):
            self.hidden_layer_sizes = [self.hidden_layer_sizes]
        self.fc_layers = [layers.Dense(layer_size) for layer_size in hidden_layer_sizes]
        self.output_layer = layers.Dense(n_outputs)
        self.optimizer = optimizers.Adam(learning_rate = lr)
        #self.optimizer = optimizers.SGD(learning_rate = lr)
        self.is_gibbs = is_gibbs
        self.L = L # If L=-1, we don't use Gaussian approximation.
        
        
    def call(self, x):
        '''
        Initial the node values and weights of the network.
        '''
        network = []
        
        for i, layer in enumerate(self.fc_layers):
            
            logits = layer(x)
            if self.L != -1: # use Gaussian approximation on HMC
                p = tfm.sigmoid(logits)
                std = tfm.sqrt(p * (1 - p) / self.L)
                x = tfp.distributions.Normal(loc = p, scale = std).sample()
            else:
                x = tfp.distributions.Bernoulli(logits = logits).sample()
                
            network.append(x)

        final_logits = self.output_layer(x) # initial the weight of output layer
            
        return network

    def get_weights(self):

        weights = []
        for layer in self.fc_layers:
            weights.append(layer.get_weights())

        weights.append(self.output_layer.get_weights())

        return weights
    
    def target_log_prob(self, x, h, y, is_kernel=False):
        '''
        Calculate the log probability of target distribution.
        '''
        # get current state
        if is_kernel: # generate hmc kernel
            h_current = tf.split(h, self.hidden_layer_sizes, axis = 1)
        else:
            h_current = [tf.cast(h_i, dtype = tf.float32) for h_i in h]

        if self.L == -1: # change value to [0,1] for HMC w/o normal approximation
            h_current = [tfm.sigmoid(t) for t in h_current]
        #else:
        #   h_current = [tf.clip_by_value(t, clip_value_min = 0.001, clip_value_max = 0.999) for t in h_current]
        h_previous = [x] + h_current[:-1]
    
        nlog_prob = 0. # negative log probability
        
        for i, (cv, pv, layer) in enumerate(zip(h_current, h_previous, self.fc_layers)):
            
            logits = layer(pv)
            if self.L != -1: # HMC with Gaussian approximation
                p = tfm.sigmoid(logits)
                pp = (self.L * p + 1) / (self.L + 2)
                std = tfm.sqrt(pp * (1 - pp) / self.L)
                layer_log_prob = normal_logpdf(cv, pp, std)

                '''
                #check p, std and layer prob
                has_nan_p = tf.reduce_any(tfm.is_nan(p))
                has_nan_p_value = has_nan_p.numpy()
                if has_nan_p_value: 
                    print("p has nan values.")

                has_zero_std = tf.reduce_any(tf.equal(std, 0.0))
                has_zero_std_value = has_zero_std.numpy()
                if has_zero_std_value:
                    print("std has zero values.")
                
                has_nan = tf.reduce_any(tfm.is_nan(layer_log_prob))
                has_nan_value = has_nan.numpy()
                if has_nan_value:
                    print("layer log prob has nan value.")
                '''
                
                nlog_prob -= tf.reduce_sum(layer_log_prob, axis=-1) #check nlog_prob
            else:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=cv, logits=logits)
                if not self.is_gibbs:
                    cross_entropy += cont_bern_log_norm(tf.nn.sigmoid(logits))
                nlog_prob += tf.reduce_sum(cross_entropy, axis=-1)
        
        if y != None:
            fce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=self.output_layer(h_current[-1]))
            nlog_prob += tf.reduce_sum(fce, axis=-1)
            
        return -1 * nlog_prob

    def gibbs_new_state(self, x, h, y):
        '''
        Generate a new state for the network node by node in Gibbs setting.
        '''
        
        h_current = h
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        
        in_layers = self.fc_layers
        out_layers = self.fc_layers[1:] + [self.output_layer]
        
        prev_vals = [x] + h_current[:-1]
        curr_vals = h_current
        next_vals = h_current[1:] + [y]
        
        for i, (in_layer, out_layer, pv, cv, nv) in enumerate(zip(in_layers, out_layers, prev_vals, curr_vals, next_vals)):

            # node by node
            
            nodes = tf.transpose(cv)
            prob_parents = tfm.sigmoid(in_layer(pv))
            
            out_layer_weights = out_layer.get_weights()[0]
            
            next_logits = out_layer(cv)
            
            new_layer = []
            
            for j, node in enumerate(nodes):
                
                # get info for current node (i, j)
                
                prob_parents_j = prob_parents[:, j]
                out_layer_weights_j = out_layer_weights[j]
                
                # calculate logits and logprob for node is 0 or 1
                next_logits_if_node_0 = next_logits[:, :] - node[:, None] * out_layer_weights_j[None, :]
                next_logits_if_node_1 = next_logits[:, :] + (1 - node[:, None]) * out_layer_weights_j[None, :]
                
                #print(next_logits_if_node_0, next_logits_if_node_1)
                
                logprob_children_if_node_0 = -1 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = tf.cast(nv, dtype = tf.float32), logits = next_logits_if_node_0), axis = -1)
                
                logprob_children_if_node_1 = -1 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = tf.cast(nv, dtype = tf.float32), logits = next_logits_if_node_1), axis = -1)
                
                # calculate prob for node (i, j)
                prob_0 = (1 - prob_parents_j) * tfm.exp(logprob_children_if_node_0)
                prob_1 = prob_parents_j * tfm.exp(logprob_children_if_node_1)
                prob_j = prob_1 / (prob_1 + prob_0)
            
                # sample new state with prob_j for node (i, j)
                new_node = tfp.distributions.Bernoulli(probs = prob_j).sample() # MAY BE SLOW
                
                # update nodes and logits for following calculation
                new_node_casted = tf.cast(new_node, dtype = "float32")
                next_logits = next_logits_if_node_0 * (1 - new_node_casted)[:, None] \
                            + next_logits_if_node_1 * new_node_casted[:, None] 
                
                # keep track of new node values (in prev/curr/next_vals and h_new)
                new_layer.append(new_node)
           
            new_layer = tf.transpose(new_layer)
            h_current[i] = new_layer
            prev_vals = [x] + h_current[:-1]
            curr_vals = h_current
            next_vals = h_current[1:] + [y]
        
        return h_current
    
    def generate_hmc_kernel(self, x, y, step_size=pow(1000, -1/4)):
        '''
        Generate the kernal of HMC.
        '''
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn = lambda v: self.target_log_prob(x, v, y, is_kernel = True),
            num_leapfrog_steps = 2,
            step_size = step_size),
            num_adaptation_steps = int(100 * 0.8))
        
        return adaptive_hmc
    

    def propose_new_state_hamiltonian(self, x, h, y, hmc_kernel, is_update_kernel=True):
        '''
        Generate a new state for the network node by node in HMC setting.
        '''
        h_current = h
        h_current = [tf.cast(h_i, dtype = tf.float32) for h_i in h_current]
        h_current = tf.concat(h_current, axis = 1)

        # run the chain (with burn-in)
        num_burnin_steps = 0
        num_results = 1

        samples = tfp.mcmc.sample_chain(
            num_results = num_results,
            num_burnin_steps = num_burnin_steps,
            current_state = h_current, # may need to be reshaped
            kernel = hmc_kernel,
            trace_fn = None,
            return_final_kernel_results = True)
    
        # Generate new states of chains
        h_state = samples[0][0]
        if self.L != -1: # clip all values to [0,1] for Gaussian approximation
            h_state = [tf.clip_by_value(t, clip_value_min = 0, clip_value_max = 1) for t in h_state]
        h_new = tf.split(h_state, self.hidden_layer_sizes, axis = 1) 
        
        # Update the kernel if necesssary
        if is_update_kernel:
            new_step_size = samples[2].new_step_size.numpy()
            ker_new = self.generate_hmc_kernel(x, y, new_step_size)
            return(h_new, ker_new)
        else:
            return h_new
    
    def update_weights(self, x, h, y):
        
        with tf.GradientTape() as tape:
            loss = -1 * tf.reduce_mean(self.target_log_prob(x, h, y))
        
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
    def predict_proba(self, x):

        logits = 0.0
        for layer in self.fc_layers:
            logits = layer(x)
            x = tfm.sigmoid(logits)
        
        logits = self.output_layer(x)
        probs = tfm.sigmoid(logits)
        return probs
    
    def get_loss(self, x, y):
        
        logits = 0.0
        for layer in self.fc_layers:
            logits = layer(x)
            x = tfm.sigmoid(logits)
            
        logits = self.output_layer(x)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.cast(y, tf.float32), logits = logits)
        return tf.reduce_sum(loss, axis = -1)