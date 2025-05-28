



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



class CustomGRULayer(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		"""
		Custom GRU layer with output transformation for single sequence element
		
		Args:
			input_size (int): Size of input features
			hidden_size (int): Size of hidden state in GRU
			output_size (int): Size of the output after transformation
		"""
		super(CustomGRULayer, self).__init__()
		
		# GRU cell for processing input
		self.gru_cell = nn.GRUCell(input_size, hidden_size)
		
		# Transformation layer to generate output from hidden state
		self.output_transform = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, output_size)
		)
		
		self.hidden_size = hidden_size
		
	def forward(self, x, h_t):
		"""
		Forward pass through the GRU layer
		
		Args:
			x (torch.Tensor): Input tensor of shape [batch_size, input_size]
			context_vector (torch.Tensor): Context vector to initialize hidden state
										 Shape: [batch_size, hidden_size]
		
		Returns:
			tuple: (output, hidden_state)
				- output: tensor of shape [batch_size, output_size]
				- hidden_state: tensor of shape [batch_size, hidden_size]
		"""
		# Initialize hidden state with context vector
		# h_t = context_vector
		
		# Update hidden state with GRU cell
		h_t = self.gru_cell(x, h_t)
		
		# Transform hidden state to generate output
		g_t = self.output_transform(h_t)
		
		return g_t, h_t


class GRU_Hidden_State(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(GRU_Hidden_State, self).__init__()
		
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

	def __init__(self, num_batch, num, t, mlp_init, inp_mean, inp_std, gru_context, gru_init):
		super(learned_qp_solver, self).__init__()

		
		self.num_batch = num_batch
		self.num = num
		self.nvar = self.num
		self.t = t
		self.mlp_init = mlp_init
		self.inp_mean = inp_mean 
		self.inp_std = inp_std
		self.gru_context = gru_context
		self.gru_init = gru_init
		
		self.t_fin = num*t 
		self.rho = 1.0
		self.rho_projection = 1.0
		self.A_projection = torch.eye(self.num, device = device)
		
		self.P_vel = torch.eye(self.num, device = device)
		self.P_pos = torch.cumsum(self.P_vel*self.t, axis = 0)
		self.P_acc = torch.diff(self.P_vel, axis = 0)/t 
		self.P_jerk = torch.diff(self.P_acc, axis = 0)/t 
		self.num_acc = self.num-1 
		self.num_jerk = self.num_acc-1
		self.maxiter = 5


		# self.A_eq = torch.vstack(( self.P_vel[0], self.P_acc[0]   ))
		self.A_eq = self.P_vel[0].reshape(1, self.num)


		self.A_vel = torch.vstack(( self.P_vel, -self.P_vel  ))
		self.A_acc = torch.vstack(( self.P_acc, -self.P_acc  ))
		self.A_jerk = torch.vstack(( self.P_jerk, -self.P_jerk  ))
		self.A_pos = torch.vstack(( self.P_pos, -self.P_pos  ))
		self.A_control = torch.vstack(( self.A_vel, self.A_acc, self.A_jerk, self.A_pos ))
		self.num_constraints = self.A_control.size(dim = 0) 
		self.rcl_loss = nn.MSELoss()


	def compute_boundary_vec(self, vel_init, acc_init):

		vel_init_vec = vel_init*torch.ones((self.num_batch, 1), device = device)
		acc_init_vec = acc_init*torch.ones((self.num_batch, 1), device = device)

		# b_eq = torch.hstack(( vel_init_vec, acc_init_vec))
		b_eq = vel_init_vec
		  
		return b_eq 
	
	def compute_s_init(self, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, vel_projected, theta_min, theta_max, theta_init):

		b_vel = torch.hstack(( vel_max*torch.ones(( self.num_batch, self.num  ), device = device), -vel_min*torch.ones(( self.num_batch, self.num  ), device = device)     ))
		b_acc = torch.hstack(( acc_max*torch.ones(( self.num_batch, self.num_acc  ), device = device), -acc_min*torch.ones(( self.num_batch, self.num_acc  ), device = device)     ))
		b_jerk = torch.hstack(( jerk_max*torch.ones(( self.num_batch, self.num_jerk  ), device = device), -jerk_min*torch.ones(( self.num_batch, self.num_jerk  ), device = device)     ))		
		b_pos = torch.hstack(( (theta_max-theta_init)*torch.ones(( self.num_batch, self.num  ), device = device), -(theta_min-theta_init)*torch.ones(( self.num_batch, self.num  ), device = device)   )   )
		b_control = torch.hstack(( b_vel, b_acc, b_jerk, b_pos  ))

		s = torch.maximum( torch.zeros(( self.num_batch, self.num_constraints ), device = device), -torch.mm(self.A_control, vel_projected.T).T+b_control  )
		
		return s
	
	def compute_feasible_control(self, vel_samples, b_eq, s, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, vel_projected, lamda, theta_min, theta_max, theta_init):
	 
		b_vel = torch.hstack(( vel_max*torch.ones(( self.num_batch, self.num  ), device = device), -vel_min*torch.ones(( self.num_batch, self.num  ), device = device)     ))
		b_acc = torch.hstack(( acc_max*torch.ones(( self.num_batch, self.num_acc  ), device = device), -acc_min*torch.ones(( self.num_batch, self.num_acc  ), device = device)     ))
		b_jerk = torch.hstack(( jerk_max*torch.ones(( self.num_batch, self.num_jerk  ), device = device), -jerk_min*torch.ones(( self.num_batch, self.num_jerk  ), device = device)     ))		
		b_pos = torch.hstack(( (theta_max-theta_init)*torch.ones(( self.num_batch, self.num  ), device = device), -(theta_min-theta_init)*torch.ones(( self.num_batch, self.num  ), device = device)   )   )
		b_control = torch.hstack(( b_vel, b_acc, b_jerk, b_pos  ))

		# s = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num+2*self.num_acc+2*self.num_jerk )), -jnp.dot(self.A_control, vel_projected.T).T+b_control  )
		

		b_control_aug = b_control-s
		
		cost = self.rho_projection*torch.mm(self.A_projection.T, self.A_projection)+self.rho*torch.mm(self.A_control.T, self.A_control)

		cost_mat = torch.vstack([torch.hstack([cost, self.A_eq.T]), torch.hstack([self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=device)])])
		lincost = -lamda-self.rho_projection*torch.mm(self.A_projection.T, vel_samples.T).T-self.rho*torch.mm(self.A_control.T, b_control_aug.T).T
	
		sol = torch.linalg.solve(cost_mat+0.00001*torch.eye(cost_mat.size(dim = 1), device = device  ), torch.hstack(( -lincost, b_eq )).T).T
		
		vel_projected = sol[:, 0: self.nvar]
  
		s = torch.maximum( torch.zeros(( self.num_batch, self.num_constraints ), device = device), -torch.mm(self.A_control, vel_projected.T).T+b_control  )
		res_vec = torch.mm(self.A_control, vel_projected.T).T-b_control+s

		res_norm = torch.linalg.norm(res_vec, dim = 1)

		lamda = lamda-self.rho*torch.mm(self.A_control.T, res_vec.T).T

		return vel_projected, s, res_norm, lamda 
	
	def project_to_boundary(self, vel_projected, b_eq):

		cost = self.rho_projection*torch.mm(self.A_projection.T, self.A_projection)

		cost_mat = torch.vstack([torch.hstack([cost, self.A_eq.T]), torch.hstack([self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=device)])])
		lincost = -self.rho_projection*torch.mm(self.A_projection.T, vel_projected.T).T
		
		sol = torch.linalg.solve(cost_mat+0.00001*torch.eye(cost_mat.size(dim = 1), device = device  ), torch.hstack(( -lincost, b_eq )).T).T
		
		vel_projected = sol[:, 0: self.nvar]
  
	

		return vel_projected
	
	def compute_projection(self, lamda, vel_projected, vel_init, acc_init,  vel_samples, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, theta_min, theta_max, theta_init, h_0, s):
	 				
		# s = self.compute_s_init(vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, vel_projected, theta_min, theta_max, theta_init) 

		b_eq = self.compute_boundary_vec(vel_init, acc_init)

		accumulated_res_primal_list = [] 

		accumulated_res_fixed_point_list= []

		h = h_0


		for i in range(0, self.maxiter):

			# vel_projected_prev = vel_projected.clone()
			lamda_prev = lamda.clone() 
			s_prev = s.clone()
			vel_projected, s, res_norm, lamda = self.compute_feasible_control(vel_samples, b_eq, s, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, vel_projected, lamda, theta_min, theta_max, theta_init)
	 
			r_1 = torch.hstack(( s_prev, lamda_prev)) 
			r_2 = torch.hstack(( s, lamda))

			r = torch.hstack(( r_1, r_2, r_2-r_1 ))
			# print(r.size())
			# print(h.size())
			gru_output, h = self.gru_context(r, h)
			s_delta = gru_output[:, 0: self.num_constraints]
			lamda_delta = gru_output[:, self.num_constraints: self.num_constraints+self.num]
			
			# vel_projected = vel_projected+vel_delta

			# vel_projected = self.project_to_boundary(vel_projected, b_eq)
			lamda = lamda+lamda_delta 
			s = s+s_delta
			s = torch.maximum( torch.zeros(( self.num_batch, self.num_constraints ), device = device), s)

			primal_residual = res_norm 
			fixed_point_residual = torch.linalg.norm(s_prev-s, dim = 1)+torch.linalg.norm(lamda_prev-lamda, dim = 1)

			accumulated_res_primal_list.append(primal_residual)
			accumulated_res_fixed_point_list.append(fixed_point_residual)

		
		res_primal_stack = torch.stack(accumulated_res_primal_list )
		res_fixed_point_stack = torch.stack(accumulated_res_fixed_point_list )
		accumulated_res_fixed_point = torch.sum(res_fixed_point_stack, dim = 0)/self.maxiter	
		accumulated_res_primal = torch.sum(res_primal_stack, dim = 0)/(self.maxiter)

		return vel_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point
	
	def decoder_function(self, inp_norm, vel_init, acc_init,  vel_samples, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, theta_min, theta_max, theta_init):
	 
		
		neural_output_batch = self.mlp_init(inp_norm)
  
		vel_projected = neural_output_batch[:, 0: self.num]
	
		lamda = neural_output_batch[:, self.num: 2*self.num]
		s = neural_output_batch[:, 2*self.num: 2*self.num+self.num_constraints]
		s = torch.maximum( torch.zeros(( self.num_batch, self.num_constraints ), device = device), s)
		h_0 = self.gru_init(inp_norm)
		
		vel_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point =  self.compute_projection(lamda, vel_projected, vel_init, acc_init, vel_samples, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, theta_min, theta_max, theta_init, h_0, s)	


		return vel_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point
	

	def ss_loss(self, accumulated_res_primal, accumulated_res_fixed_point, vel_projected, vel_samples):
	 
	
		primal_loss = (0.5 * (torch.mean(accumulated_res_primal)))
		fixed_point_loss = (0.5 * (torch.mean(accumulated_res_fixed_point)))
		
		proj_loss = 0.5*self.rcl_loss(vel_projected, vel_samples)
  		
		# loss = fixed_point_loss+primal_loss+proj_loss
		loss = fixed_point_loss+primal_loss+proj_loss


		# loss = primal_loss+proj_loss
		# loss = fixed_point_loss
 
		return loss, primal_loss, fixed_point_loss, proj_loss
	


	def forward(self, inp, vel_init, acc_init, vel_samples, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, theta_min, theta_max, theta_init):

		# Normalize input
		inp_norm = (inp - self.inp_mean) / self.inp_std

  
		vel_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point = self.decoder_function(inp_norm, vel_init, acc_init,  vel_samples, vel_max, vel_min, acc_max, acc_min, jerk_max, jerk_min, theta_min, theta_max, theta_init)
	 
	
		return vel_projected, res_primal_stack, res_fixed_point_stack, accumulated_res_primal, accumulated_res_fixed_point
	



	

		







	
	
		



