





import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import scipy

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)


# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class MLP_Init(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(MLP_Init, self).__init__()
		
		# MC Dropout Architecture
		self.mlp = nn.Sequential(
			nn.Linear(inp_dim, hidden_dim),
			# nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			# nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, out_dim),
		)
	
	def forward(self, x):
		out = self.mlp(x)
		return out
	


class learned_qp_solver(nn.Module):

	def __init__(self, num_batch, num, t, mlp_init):
		super(learned_qp_solver, self).__init__()

		
		self.num_batch = num_batch
		self.num = num
		self.nvar = self.num
		self.t = t
		self.mlp_init = mlp_init
		# self.inp_mean = inp_mean 
		# self.inp_std = inp_std
		
		self.t_fin = num*t 
		self.rho = 1.0
		self.rho_projection = 1.0
		self.A_projection = torch.eye(self.num, device = device)
		self.nvar = self.num
		
		self.P_steer = torch.eye(self.num, device = device)
		self.P_steerdot = torch.diff(self.P_steer, axis = 0)/t 
		self.P_steerddot = torch.diff(self.P_steerdot, axis = 0)/t 
		self.num_steerdot = self.num-1 
		self.num_steerddot = self.num_steerdot-1
		self.maxiter = 5
		self.A_eq = self.P_steer[0].reshape(1, self.num)


		self.A_steer = torch.vstack(( self.P_steer, -self.P_steer  ))
		self.A_steerdot = torch.vstack(( self.P_steerdot, -self.P_steerdot  ))
		self.A_steerddot = torch.vstack(( self.P_steerddot, -self.P_steerddot  ))
		self.A_control = torch.vstack(( self.A_steer, self.A_steerdot, self.A_steerddot))
		self.num_constraints = self.A_control.size(dim = 0) 
		
		
		self.rcl_loss = nn.MSELoss()


	def compute_boundary_vec(self, steer_init):

		steer_init_vec = steer_init*torch.ones((self.num_batch, 1), device = device)
		b_eq = steer_init_vec
		  
		return b_eq 
	
	
	def compute_feasible_control(self, steer_samples, b_eq, s, steer_max, steer_min, steerdot_max, steerdot_min, steerddot_max, steerddot_min, lamda):
	 
		b_vel = torch.hstack(( steer_max*torch.ones(( self.num_batch, self.num  ), device = device), -steer_min*torch.ones(( self.num_batch, self.num  ), device = device)     ))
		b_acc = torch.hstack(( steerdot_max*torch.ones(( self.num_batch, self.num_steerdot  ), device = device), -steerdot_min*torch.ones(( self.num_batch, self.num_steerdot  ), device = device)     ))
		b_jerk = torch.hstack(( steerddot_max*torch.ones(( self.num_batch, self.num_steerddot  ), device = device), -steerddot_min*torch.ones(( self.num_batch, self.num_steerddot  ), device = device)     ))		
		b_control = torch.hstack(( b_vel, b_acc, b_jerk))

		b_control_aug = b_control-s
		
		cost = self.rho_projection*torch.mm(self.A_projection.T, self.A_projection)+self.rho*torch.mm(self.A_control.T, self.A_control)

		cost_mat = torch.vstack([torch.hstack([cost, self.A_eq.T]), torch.hstack([self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=device)])])
		lincost = -lamda-self.rho_projection*torch.mm(self.A_projection.T, steer_samples.T).T-self.rho*torch.mm(self.A_control.T, b_control_aug.T).T
	
		sol = torch.linalg.solve(cost_mat, torch.hstack(( -lincost, b_eq )).T).T
		
		steer_projected = sol[:, 0: self.nvar]
  
		s = torch.maximum( torch.zeros(( self.num_batch, self.num_constraints ), device = device), -torch.mm(self.A_control, steer_projected.T).T+b_control  )
		res_vec = torch.mm(self.A_control, steer_projected.T).T-b_control+s

		res_norm = torch.linalg.norm(res_vec, dim = 1)

		lamda = lamda-self.rho*torch.mm(self.A_control.T, res_vec.T).T

		return steer_projected, s, res_norm, lamda 
	
	def compute_projection(self, lamda, steer_init, steer_samples, steer_max, steer_min, steerdot_max, steerdot_min, steerddot_max, steerddot_min, s):
	 				
		b_eq = self.compute_boundary_vec(steer_init)

		accumulated_res_primal_list = [] 

		accumulated_res_fixed_point_list= []


		for i in range(0, self.maxiter):

			lamda_prev = lamda.clone() 
			s_prev = s.clone()
			steer_projected, s, res_norm, lamda = self.compute_feasible_control( steer_samples, b_eq, s, steer_max, steer_min, steerdot_max, steerdot_min, steerddot_max, steerddot_min, lamda)
	 
			primal_residual = res_norm 
			fixed_point_residual = torch.linalg.norm(s_prev-s, dim = 1)+torch.linalg.norm(lamda_prev-lamda, dim = 1)
			

			accumulated_res_primal_list.append(primal_residual)
			accumulated_res_fixed_point_list.append(fixed_point_residual)

		
		res_primal_stack = torch.stack(accumulated_res_primal_list )
		res_fixed_point_stack = torch.stack(accumulated_res_fixed_point_list )
		accumulated_res_fixed_point = torch.sum(res_fixed_point_stack, dim = 0)/self.maxiter	
		accumulated_res_primal = torch.sum(res_primal_stack, dim = 0)/(self.maxiter)

		return steer_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point

	
	def decoder_function(self, inp_norm, steer_init, steer_samples, steer_max, steer_min, steerdot_max, steerdot_min, steerddot_max, steerddot_min):
	 
		
		neural_output_batch = self.mlp_init(inp_norm)
  
		lamda = neural_output_batch[:, 0: self.nvar]

		s = neural_output_batch[:, self.nvar: self.nvar+self.num_constraints]
		
		s = torch.maximum( torch.zeros(( self.num_batch, self.num_constraints ), device = device), s)
		
		steer_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point =  self.compute_projection(lamda, steer_init, steer_samples, steer_max, steer_min, steerdot_max, steerdot_min, steerddot_max, steerddot_min, s)	


		return steer_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point
	

	def ss_loss(self, accumulated_res_primal, accumulated_res_fixed_point, steer_projected, steer_samples):
	 
	
		primal_loss = (0.5 * (torch.mean(accumulated_res_primal)))
		fixed_point_loss = (0.5 * (torch.mean(accumulated_res_fixed_point)))
		
		proj_loss = 0.5*self.rcl_loss(steer_projected, steer_samples)
  		
		loss = fixed_point_loss+primal_loss+proj_loss
		
		return loss, primal_loss, fixed_point_loss, proj_loss
	


	def forward(self, inp, steer_init, steer_samples, steer_max, steer_min, steerdot_max, steerdot_min, steerddot_max, steerddot_min, median_, iqr_):

		# Normalize input
		# inp_norm = (inp - self.inp_mean) / self.inp_std
		inp_norm = (inp-median_)/(iqr_)
		inp_norm = inp

  
		steer_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point = self.decoder_function(inp_norm, steer_init, steer_samples, steer_max, steer_min, steerdot_max, steerdot_min, steerddot_max, steerddot_min)
	 
	
		return steer_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point
	



	

		







	
	
		



