


import numpy as np
import jax.numpy as jnp
from jax import random, vmap, jit

from functools import partial
import scipy
import matplotlib.pyplot as plt
import time

from scipy.io import loadmat
import jax.lax as lax
import jax

# jax.config.update("jax_enable_x64", True)

class qp_solver():

	def __init__(self, num_batch, num, t):
		super(qp_solver, self).__init__()

		
		self.num_batch = num_batch
		self.num = num
		self.nvar = self.num
		self.t = t
		
		self.t_fin = num*t 
		self.rho = 1.0
		self.rho_projection = 1.0
		self.A_projection = jnp.identity(self.num)
		
		self.P_vel = jnp.identity(self.num)
		self.P_pos = jnp.cumsum(self.P_vel*self.t, axis = 0)
		self.P_acc = jnp.diff(self.P_vel, axis = 0)/t 
		self.P_jerk = jnp.diff(self.P_acc, axis = 0)/t 
		self.num_acc = self.num-1 
		self.num_jerk = self.num_acc-1
		self.maxiter = 1000


		

		self.A_eq = self.P_vel[0].reshape(1, self.nvar)


		self.A_vel = jnp.vstack(( self.P_vel, -self.P_vel  ))
		self.A_acc = jnp.vstack(( self.P_acc, -self.P_acc  ))
		self.A_jerk = jnp.vstack(( self.P_jerk, -self.P_jerk  ))
		self.A_pos = jnp.vstack(( self.P_pos, -self.P_pos  ))
		self.A_control = jnp.vstack(( self.A_vel, self.A_acc, self.A_jerk, self.A_pos ))

		self.num_constraints = jnp.shape(self.A_control)[0]

	
	@partial(jit, static_argnums=(0,))	
	def compute_boundary_vec(self, vel_init):

		vel_init_vec = vel_init*jnp.ones((self.num_batch, 1))

		b_eq = vel_init_vec
		  
		return b_eq 
	

	@partial(jit, static_argnums=(0,))
	def compute_s_init(self, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, vel_projected, theta_min, theta_max, theta_init):

		b_vel = jnp.hstack(( vel_max*jnp.ones(( self.num_batch, self.num  )), -vel_min*jnp.ones(( self.num_batch, self.num  ))     ))
		b_acc = jnp.hstack(( acc_max*jnp.ones(( self.num_batch, self.num_acc  )), -acc_min*jnp.ones(( self.num_batch, self.num_acc  ))     ))
		b_jerk = jnp.hstack(( jerk_max*jnp.ones(( self.num_batch, self.num_jerk  )), -jerk_min*jnp.ones(( self.num_batch, self.num_jerk  ))     ))		
		b_pos = jnp.hstack(( (theta_max-theta_init)*jnp.ones(( self.num_batch, self.num  )), -(theta_min-theta_init)*jnp.ones(( self.num_batch, self.num  ))   )   )
		b_control = jnp.hstack(( b_vel, b_acc, b_jerk, b_pos  ))

		s = jnp.maximum( jnp.zeros(( self.num_batch, self.num_constraints )), -jnp.dot(self.A_control, vel_projected.T).T+b_control  )
		
		return s


	@partial(jit, static_argnums=(0,))
	def compute_feasible_control(self, vel_samples, b_eq, s, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, vel_projected, lamda, theta_min, theta_max, theta_init):
	 
		b_vel = jnp.hstack(( vel_max*jnp.ones(( self.num_batch, self.num  )), -vel_min*jnp.ones(( self.num_batch, self.num  ))     ))
		b_acc = jnp.hstack(( acc_max*jnp.ones(( self.num_batch, self.num_acc  )), -acc_min*jnp.ones(( self.num_batch, self.num_acc  ))     ))
		b_jerk = jnp.hstack(( jerk_max*jnp.ones(( self.num_batch, self.num_jerk  )), -jerk_min*jnp.ones(( self.num_batch, self.num_jerk  ))     ))		
		b_pos = jnp.hstack(( (theta_max-theta_init)*jnp.ones(( self.num_batch, self.num  )), -(theta_min-theta_init)*jnp.ones(( self.num_batch, self.num  ))   )   )
		b_control = jnp.hstack(( b_vel, b_acc, b_jerk, b_pos  ))

		# s = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num+2*self.num_acc+2*self.num_jerk )), -jnp.dot(self.A_control, vel_projected.T).T+b_control  )
		

		b_control_aug = b_control-s
		
		cost = jnp.dot(self.A_projection.T, self.A_projection)+self.rho*jnp.dot(self.A_control.T, self.A_control)

		cost_mat = jnp.vstack((  jnp.hstack(( cost, self.A_eq.T )), jnp.hstack(( self.A_eq, jnp.zeros(( jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0] )) )) ))
		lincost = -lamda-jnp.dot(self.A_projection.T, vel_samples.T).T-self.rho*jnp.dot(self.A_control.T, b_control_aug.T).T
	
		sol = jnp.linalg.solve(cost_mat, jnp.hstack(( -lincost, b_eq )).T).T
		
		vel_projected = sol[:, 0: self.nvar]
  
		s = jnp.maximum( jnp.zeros(( self.num_batch, self.num_constraints )), -jnp.dot(self.A_control, vel_projected.T).T+b_control  )
		res_vec = jnp.dot(self.A_control, vel_projected.T).T-b_control+s

		res_norm = jnp.linalg.norm(res_vec, axis = 1)

		lamda = lamda-self.rho*jnp.dot(self.A_control.T, res_vec.T).T

		return vel_projected, s, res_norm, lamda 
	

	@partial(jit, static_argnums=(0,))
	def compute_projection(self, lamda, vel_projected, vel_init, vel_samples, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, theta_min, theta_max, theta_init, s):
	 		

		vel_projected_init = vel_projected 
		lamda_init = lamda 

		# s_init = self.compute_s_init(vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, vel_projected, theta_min, theta_max, theta_init) 
		s_init = s

		b_eq = self.compute_boundary_vec(vel_init)

		
  
		def lax_custom_projection(carry, idx):
	  
			vel_projected, lamda, s = carry
			vel_projected_prev = vel_projected 
			lamda_prev = lamda 
			s_prev = s
	  
			vel_projected, s, res_norm, lamda = self.compute_feasible_control(vel_samples, b_eq, s, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, vel_projected, lamda, theta_min, theta_max, theta_init)
	 
			primal_residual = res_norm 
			# fixed_point_residual = jnp.linalg.norm(vel_projected_prev-vel_projected, axis = 1)+jnp.linalg.norm(lamda_prev-lamda_prev, axis = 1)
			fixed_point_residual = jnp.linalg.norm(s_prev-s, axis = 1)+jnp.linalg.norm(lamda_prev-lamda_prev, axis = 1)
			

			
			return (vel_projected, lamda, s), (primal_residual, fixed_point_residual)		
  
		carry_init = (vel_projected_init, lamda_init, s_init )
		carry_final, res_tot = lax.scan(lax_custom_projection, carry_init, jnp.arange(self.maxiter))

		vel_projected, lamda, s = carry_final
	  
		primal_residual, fixed_point_residual = res_tot
  
		return vel_projected, primal_residual, fixed_point_residual 


		
	



	





		



	


  
