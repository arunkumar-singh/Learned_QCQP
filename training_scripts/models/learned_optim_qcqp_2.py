

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import scipy

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PointNet(nn.Module):
	def __init__(self, inp_channel, emb_dims, output_channels=20):
		super(PointNet, self).__init__()
		self.conv1 = nn.Conv1d(inp_channel, 64, kernel_size=1, bias=False) # input_channel = 3
		self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
		self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(emb_dims)
		self.linear1 = nn.Linear(emb_dims, 256, bias=False)
		self.bn6 = nn.BatchNorm1d(256)
		self.dp1 = nn.Dropout()
		self.linear2 = nn.Linear(256, output_channels)
	
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
		x = F.relu(self.bn6(self.linear1(x)))
		x = self.dp1(x)
		x = self.linear2(x)
		return x



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
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, out_dim),
		)
	
	def forward(self, x):
		out = self.mlp(x)
		return out



class MLP_Pred(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(MLP_Pred, self).__init__()
		
		# MC Dropout Architecture
		self.mlp = nn.Sequential(
			nn.Linear(inp_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
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
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, out_dim),
		)
	
	def forward(self, x):
		out = self.mlp(x)
		return out
	

class Learned_QCQP(nn.Module):
	
	def __init__(self, num_obs, t_fin, P, Pdot, Pddot, point_net, num_batch, min_pcd, max_pcd, inp_mean, inp_std, gru_context, gru_hidden_state_init, mlp_init):
		super(Learned_QCQP, self).__init__()
		
		# BayesMLP
		
		self.point_net = point_net 
		self.mlp_init = mlp_init
		#self.mlp_2 = mlp_2
  
		self.gru_context = gru_context 
		self.gru_hidden_state_init = gru_hidden_state_init

		self.min_pcd = min_pcd 
		self.max_pcd = max_pcd 

		# Normalizing Constants
		self.inp_mean = inp_mean
		self.inp_std = inp_std
		self.num_batch = num_batch

		# P Matrices
		self.P = P.to(device)
		self.Pdot = Pdot.to(device)
		self.Pddot = Pddot.to(device)


		self.A_eq_x = torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1]    ]  )
		self.A_eq_y = torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1]     ]  )

		self.A_eq = torch.block_diag(self.A_eq_x, self.A_eq_y)
  
				
		# No. of Variables
		self.nvar = P.size(dim = 1)
		self.num = P.size(dim = 0)
		self.num_batch = num_batch
  
		self.A_projection = torch.eye(self.nvar, device = device)

		# self.a_obs = 6.00
		# self.b_obs = 3.20
		self.a_ego = 3.0 
		self.b_ego = 1.5 
		self.margin_longitudinal = 0.2
		self.margin_lateral = 0.2

		self.wheel_base = 2.5
		self.steer_max = 0.5
		self.kappa_max = 0.3

  
		self.num_circles = 3
		self.num_obs = num_obs
		
		# Parameters
		  
		self.rho_obs = 1.0
		self.rho_ineq = 1.0
		self.rho_projection = 1
		self.rho_lane = 1
		self.rho_projection = 1

		self.weight_smoothness = 200

		# self.cost_smoothness = self.weight_smoothness *( torch.mm(self.Pddot.T, self.Pddot))#+0.1*torch.eye(self.nvar, device = device))
		# self.cost_smoothness = self.weight_smoothness *( torch.mm(self.Pddot.T, self.Pddot)+0.1*torch.eye(self.nvar, device = device))
  
  
		self.cost_smoothness = self.weight_smoothness *( torch.eye(self.nvar, device = device))
		self.weight_v_des = 2
		self.weight_y_des = 2
  
		
		self.v_min = 0.001
		self.v_max = 18

		self.a_max = 6.0
		 
		self.A_obs = torch.tile(self.P, (self.num_obs*self.num_circles, 1))
		
		self.A_vel = self.Pdot
		self.A_acc = self.Pddot
		self.A_projection = torch.eye(self.nvar, device = device)
		self.A_lane = torch.vstack((self.P, -self.P))
		
		self.A_v_des = self.Pdot   
		self.A_y_des = self.P

		self.maxiter = 5 # 20
  
		self.t_fin = t_fin
		
		self.tot_time = torch.linspace(0, t_fin, self.num, device=device)
		########################################
		
  		# RCL Loss
		self.rcl_loss = nn.MSELoss()
		self.soft = nn.Softplus()
		self.compute_obs_trajectories_batch = torch.vmap(self.compute_obs_trajectories, in_dims = (0, 0, 0, 0 )  )
		self.compute_obs_ellipse_batch = torch.vmap(self.compute_obs_ellipse, in_dims = (0, 0 )  )
		self.compute_x_batch = torch.vmap(self.compute_x, in_dims = (0, 0, 0, 0 )  )
		


	def compute_boundary_layer_optim(self, init_state_ego, goal_des):
	 
		x_init_vec = init_state_ego[:, 0].reshape(self.num_batch, 1) 
		y_init_vec = init_state_ego[:, 1].reshape(self.num_batch, 1) 
  
		vx_init_vec = init_state_ego[:, 2].reshape(self.num_batch, 1)
		vy_init_vec = init_state_ego[:, 3].reshape(self.num_batch, 1)
  
		ax_init_vec = init_state_ego[:, 4].reshape(self.num_batch, 1)
		ay_init_vec = init_state_ego[:, 5].reshape(self.num_batch, 1)
  
		x_fin_vec = goal_des[:, 0].reshape(self.num_batch, 1)
		y_fin_vec = goal_des[:, 1].reshape(self.num_batch, 1)
  
		vy_fin_vec = torch.zeros(( self.num_batch, 1   ), device = device)
  
		b_eq_x = torch.hstack([x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec  ])
		b_eq_y = torch.hstack([y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec, vy_fin_vec  ])
	
		return b_eq_x, b_eq_y	
	
	def compute_obs_trajectories(self, closest_obs, psi_obs, dim_x_obs, dim_y_obs):
	 
	 
		x_obs = closest_obs[0 : self.num_obs]
		y_obs = closest_obs[self.num_obs : 2*self.num_obs]
	 
		vx_obs = torch.zeros(  self.num_obs, device = device)
		vy_obs = torch.zeros(  self.num_obs, device = device)
		
		x_obs_traj = (x_obs + vx_obs * self.tot_time[:, None]).T # num_obs x num
		y_obs_traj = (y_obs + vy_obs * self.tot_time[:, None]).T
		
		psi_obs_traj = torch.tile(psi_obs, (self.num,1)).T # num_obs x num
		
		x_obs_circles, y_obs_circles, psi_obs_circles = self.split(x_obs_traj, y_obs_traj, psi_obs_traj, dim_x_obs, dim_y_obs)
		x_obs_circles = x_obs_circles.reshape(self.num_obs*self.num*self.num_circles)
		y_obs_circles = y_obs_circles.reshape(self.num_obs*self.num*self.num_circles)
		psi_obs_circles = psi_obs_circles.reshape(self.num_obs*self.num*self.num_circles)
		
		

		return x_obs_circles.float(), y_obs_circles.float(), psi_obs_circles.float()
 
	def split(self, x_obs_traj, y_obs_traj, psi_obs_traj, dim_x_obs, dim_y_obs):
	 
		dist_centre = self.compute_centres(dim_x_obs, dim_y_obs) # num_obs x num_circles x num
				
		psi_obs_traj = torch.tile(psi_obs_traj,(self.num_circles,)).reshape(self.num_obs,self.num_circles,-1)
		x_obs_traj = torch.tile(x_obs_traj,(self.num_circles,)).reshape(self.num_obs,self.num_circles,-1)
		y_obs_traj = torch.tile(y_obs_traj,(self.num_circles,)).reshape(self.num_obs,self.num_circles,-1)

		x_temp = dist_centre*torch.cos(psi_obs_traj)
		y_temp = dist_centre*torch.sin(psi_obs_traj)

		x_obs_circles = x_obs_traj + x_temp
		y_obs_circles = y_obs_traj + y_temp

		x_obs_circles = x_obs_circles.reshape(self.num_circles*self.num_obs,-1)
		y_obs_circles = y_obs_circles.reshape(self.num_circles*self.num_obs,-1)
	   
		return x_obs_circles,y_obs_circles,psi_obs_traj.reshape(self.num_circles*self.num_obs,-1)

	def compute_centres(self,dim_x_obs, dim_y_obs):
	 
		r1 = torch.zeros(self.num_obs, device= device)
		r2 = (dim_x_obs-dim_y_obs)/2 
		r3 = -(dim_x_obs-dim_y_obs)/2 

		dist_centre = torch.vstack((r1, r2, r3)).T # num_obs x num_circles
		# dist_centre = torch.tile(dist_centre,(self.num,))
  
		# print(dist_centre.size())
  
		dist_centre = torch.tile(dist_centre,(self.num,)).reshape(self.num_obs,self.num,-1).permute(0,2,1) # num_obs x num_circles x num
  
		return dist_centre
	
	def compute_obs_ellipse(self,dim_x_obs, dim_y_obs):
		radius = dim_y_obs/2 
		a_obs_1 = self.a_ego + radius + self.margin_longitudinal
		b_obs_1 = self.b_ego + radius + self.margin_lateral

		a_obs = torch.tile(a_obs_1,(self.num_circles,1)).T.reshape(self.num_circles*self.num_obs)
		b_obs = torch.tile(b_obs_1,(self.num_circles,1)).T.reshape(self.num_circles*self.num_obs)

		a_obs_1 = a_obs
		b_obs_1 = b_obs

		a_obs = torch.tile(a_obs,(self.num,1)).T.reshape(self.num_circles*self.num_obs*self.num)
		b_obs = torch.tile(b_obs,(self.num,1)).T.reshape(self.num_circles*self.num_obs*self.num)

		return a_obs,b_obs



	def compute_alph_d(self, primal_sol, x_obs_circle_traj, y_obs_circle_traj, y_ub, y_lb, a_obs, b_obs ):
	 
		primal_sol_x = primal_sol[:, 0:self.nvar]
		primal_sol_y = primal_sol[:, self.nvar:2 * self.nvar]	

		x = torch.mm(self.P, primal_sol_x.T).T
		xdot = torch.mm(self.Pdot, primal_sol_x.T).T 
		xddot = torch.mm(self.Pddot, primal_sol_x.T).T
  
		y = torch.mm(self.P, primal_sol_y.T).T
		ydot = torch.mm(self.Pdot, primal_sol_y.T).T
		yddot = torch.mm(self.Pddot, primal_sol_y.T).T

		########################################################## Obstacle update
  
		x_extend = torch.tile(x, (1, self.num_obs*self.num_circles))
		y_extend = torch.tile(y, (1, self.num_obs*self.num_circles))

		wc_alpha = (x_extend - x_obs_circle_traj)
		ws_alpha = (y_extend - y_obs_circle_traj)

		wc_alpha = wc_alpha.reshape(self.num_batch, self.num * self.num_obs*self.num_circles)
		ws_alpha = ws_alpha.reshape(self.num_batch, self.num * self.num_obs*self.num_circles)
  
		alpha_obs = torch.atan2(ws_alpha * a_obs, wc_alpha * b_obs)
		c1_d = 1.0 * self.rho_obs*(a_obs**2 * torch.cos(alpha_obs)**2 + b_obs**2 * torch.sin(alpha_obs)**2)
		c2_d = 1.0 * self.rho_obs*(a_obs * wc_alpha * torch.cos(alpha_obs) + b_obs * ws_alpha * torch.sin(alpha_obs))
  
		d_temp = c2_d/c1_d
		d_obs = torch.maximum(torch.ones((self.num_batch, self.num * self.num_obs*self.num_circles), device=device), d_temp)
  

		
		###############################################33
		wc_alpha_vx = xdot
		ws_alpha_vy = ydot

		alpha_v = torch.atan2( ws_alpha_vy, wc_alpha_vx)
		alpha_v = torch.clip(alpha_v, -torch.pi/3*torch.ones(( self.num_batch, self.num  ), device = device), torch.pi/3*torch.ones(( self.num_batch, self.num  ), device = device)   )
		
		c1_d_v = 1.0 * self.rho_ineq * (torch.cos(alpha_v)**2 + torch.sin(alpha_v)**2)
		c2_d_v = 1.0 * self.rho_ineq * (wc_alpha_vx * torch.cos(alpha_v) + ws_alpha_vy * torch.sin(alpha_v))
		
		d_temp_v = c2_d_v/c1_d_v

		d_v = torch.clip(d_temp_v,  torch.tensor(self.v_min).to(device), torch.tensor(self.v_max).to(device))
	
	
		#####################################################################
		
		wc_alpha_ax = xddot
		ws_alpha_ay = yddot
		alpha_a = torch.atan2( ws_alpha_ay, wc_alpha_ax)

		c1_d_a = 1.0 * self.rho_ineq * (torch.cos(alpha_a)**2 + torch.sin(alpha_a)**2)
		c2_d_a = 1.0 * self.rho_ineq * (wc_alpha_ax * torch.cos(alpha_a) + ws_alpha_ay * torch.sin(alpha_a))

		d_temp_a = c2_d_a/c1_d_a
		# a_max_aug = (d_v**2)*(self.kappa_max)/(torch.abs(torch.sin(alpha_a-alpha_v) )+0.00001)

		d_a = torch.clip(d_temp_a, torch.zeros((self.num_batch, self.num), device=device), torch.tensor(self.a_max).to(device) )
  
  
  
		############################### Lane 
  
		# Extending Dimension
		y_ub = y_ub[:, None]
		y_lb = y_lb[:, None]
  
		b_lane = torch.hstack(( y_ub * torch.ones((self.num_batch, self.num), device=device), -y_lb * torch.ones((self.num_batch, self.num), device=device)))

  
		s_lane = torch.maximum( torch.zeros((self.num_batch, 2 * self.num), device=device), -torch.mm(self.A_lane, primal_sol_y.T).T + b_lane)
		res_lane_vec = torch.mm(self.A_lane, primal_sol_y.T).T - b_lane + s_lane
  
		curvature = (xddot*ydot-xdot*yddot)/((xdot**2+ydot**2+0.0001)**(1.5))

		steer = torch.arctan(curvature*self.wheel_base)
  
		steer_penalty_ub = torch.maximum( torch.zeros(( self.num_batch, self.num  ), device = device), steer-self.steer_max*torch.ones((self.num_batch, self.num), device = device        )       ) 
		steer_penalty_lb = torch.maximum( torch.zeros(( self.num_batch, self.num  ), device = device), -steer-self.steer_max*torch.ones((self.num_batch, self.num), device = device        )       ) 

		steer_penalty = steer_penalty_lb+steer_penalty_ub

		#########################################3

		res_ax_vec = xddot - d_a * torch.cos(alpha_a)
		res_ay_vec = yddot - d_a * torch.sin(alpha_a)
		
		res_vx_vec = xdot - d_v * torch.cos(alpha_v)
		res_vy_vec = ydot - d_v * torch.sin(alpha_v) 
  
		res_x_obs_vec = wc_alpha - a_obs * d_obs * torch.cos(alpha_obs)
		res_y_obs_vec = ws_alpha - b_obs * d_obs * torch.sin(alpha_obs)
  
		res_vel_vec = torch.hstack([res_vx_vec,  res_vy_vec])
		res_acc_vec = torch.hstack([res_ax_vec,  res_ay_vec])
		res_obs_vec = torch.hstack([res_x_obs_vec, res_y_obs_vec   ])
		
  
  
		res_norm_batch = torch.linalg.norm(res_acc_vec, dim=1) + \
						 torch.linalg.norm(res_vel_vec, dim=1) +torch.linalg.norm(res_obs_vec, axis = 1)+torch.linalg.norm(res_lane_vec, axis = 1)+torch.linalg.norm(steer_penalty, dim =1)
	   				   
		return res_norm_batch
	
	
	
	def compute_x(self, b_eq_x, b_eq_y, A_mat_vec, b_vec):
     
     
		b_eq = torch.hstack(( b_eq_x, b_eq_y  ))
		
  
		A_mat = A_mat_vec.reshape(2*self.nvar, 2*self.nvar)	
  
		cost_smoothness = torch.block_diag(self.cost_smoothness, self.cost_smoothness  )
  
		cost = cost_smoothness+torch.mm(A_mat.T, A_mat)
  
		# lincost = -torch.mm(A_mat.T, b_vec.reshape(2*self.nvar, 1))
  
		lincost = -A_mat.T @ b_vec
  
      
		cost_mat = torch.vstack([torch.hstack([cost, self.A_eq.T]), torch.hstack([self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=device)])])
		
		sol = torch.linalg.solve(cost_mat, torch.hstack(( -lincost, b_eq )))
		
		primal_sol = sol[0: 2*self.nvar]

		# primal_sol_x = primal_sol[0:self.nvar]
		# primal_sol_y = primal_sol[self.nvar:2*self.nvar]

		return primal_sol
	
	def custom_forward(self, init_state_ego, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, y_ub, y_lb, goal_des, A_mat_vec, b_vec, h_0):	

		h = h_0
		accumulated_res_primal_list = [] 
		
		# # Boundary conditions
		b_eq_x, b_eq_y = self.compute_boundary_layer_optim(init_state_ego, goal_des)

		for i in range(0, self.maxiter):
	 
			primal_sol = self.compute_x_batch( b_eq_x, b_eq_y, A_mat_vec, b_vec)

			
	
			res_norm_batch = self.compute_alph_d(primal_sol, x_obs_circle_traj, y_obs_circle_traj, y_ub, y_lb, a_obs, b_obs)

			r = torch.hstack(( primal_sol,  res_norm_batch.reshape(self.num_batch, 1)  ))
			gru_output, h = self.gru_context(r, h)
			A_mat_vec = gru_output[:, 0: (2*self.nvar)**2]
			b_vec = gru_output[:, (2*self.nvar)**2: (2*self.nvar)**2+2*self.nvar]

			accumulated_res_primal_list.append(res_norm_batch)

		
		res_primal_stack = torch.stack(accumulated_res_primal_list )
		
  
		accumulated_res_primal = torch.sum(res_primal_stack, dim = 0)/(self.maxiter)
			

		return 	primal_sol, accumulated_res_primal, res_primal_stack
	

	
	def decoder_function(self, inp_norm, init_state_ego, pcd_scaled, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, y_ub, y_lb, param_des, goal_des):
	 
		# PCD feature extractor
		pcd_features = self.point_net(pcd_scaled)

		inp_features = torch.cat([inp_norm, pcd_features], dim = 1)

		neural_output_batch = self.mlp_init(inp_features)
  
		A_mat_vec = neural_output_batch[:, 0: (2*self.nvar)**2]
		b_vec = neural_output_batch[:, (2*self.nvar)**2: (2*self.nvar)**2+2*self.nvar]

		h_0 = self.gru_hidden_state_init(inp_features)

		primal_sol, accumulated_res_primal, res_primal_stack =  self.custom_forward(init_state_ego, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, y_ub, y_lb, goal_des, A_mat_vec, b_vec, h_0)	
		
		return primal_sol, accumulated_res_primal, res_primal_stack
	


	def ss_loss(self, accumulated_res_primal, primal_sol):
	 
	
		primal_loss = (0.5 * (torch.mean(accumulated_res_primal)))
  		
		loss = primal_loss
 
		return loss


	def forward(self, inp, init_state_ego, param_des, pcd_data,  closest_obs, psi_obs, dim_x_obs, dim_y_obs,  y_ub, y_lb, P_diag, Pddot_diag, pcd_mean, pcd_std, goal_des):

		# Normalize input
		inp_norm = (inp - self.inp_mean) / self.inp_std
  
		# Batch Trajectory Prediction
		x_obs_circle_traj, y_obs_circle_traj, psi_obs_circle_traj = self.compute_obs_trajectories_batch(closest_obs, psi_obs, dim_x_obs, dim_y_obs)
		a_obs, b_obs = self.compute_obs_ellipse_batch(dim_x_obs, dim_y_obs)
  
		pcd_scaled = (pcd_data - self.min_pcd) / (self.max_pcd - self.min_pcd)

		# pcd_scaled = (pcd_data - pcd_mean) / (self.max_pcd - pcd_std)
				
		# Decode y
		primal_sol, accumulated_res_primal, res_primal_stack= self.decoder_function(inp_norm, init_state_ego, pcd_scaled, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, y_ub, y_lb, param_des, goal_des)
	 
		# predict_traj = (P_diag @ primal_sol.T).T
		# predict_acc = (Pddot_diag @ primal_sol.T).T

		return primal_sol, accumulated_res_primal
	







		



