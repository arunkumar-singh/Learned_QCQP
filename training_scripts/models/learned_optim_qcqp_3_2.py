





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
			nn.LayerNorm(hidden_size),
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
			nn.LeakyReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			# nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(),
			
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
			# nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			# nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(),
			
			nn.Linear(hidden_dim, out_dim),
		)
	
	def forward(self, x):
		out = self.mlp(x)
		return out
	

class Learned_QCQP(nn.Module):
	
	def __init__(self, num_obs, t_fin, P, Pdot, Pddot, point_net, num_batch, min_pcd, max_pcd, inp_mean, inp_std, gru_context_obs, gru_hidden_state_init_obs, gru_context_acc, gru_hidden_state_init_acc, gru_context_vel, gru_hidden_state_init_vel,  gru_context_slane, gru_hidden_state_init_slane, gru_context_lamda, gru_hidden_state_init_lamda, mlp_init_obs, mlp_init_acc, mlp_init_vel, mlp_init_slane, mlp_init_lamda):
		super(Learned_QCQP, self).__init__()
		
		# BayesMLP
		
		self.point_net = point_net 
		self.mlp_init_obs = mlp_init_obs 
		self.mlp_init_acc = mlp_init_acc 
		self.mlp_init_vel = mlp_init_vel 
		self.mlp_init_slane = mlp_init_slane  
		self.mlp_init_lamda = mlp_init_lamda 
		
		#self.mlp_2 = mlp_2
  
		self.gru_context_obs = gru_context_obs 
		self.gru_hidden_state_init_obs = gru_hidden_state_init_obs

		self.gru_context_acc = gru_context_acc 
		self.gru_hidden_state_init_acc = gru_hidden_state_init_acc

		self.gru_context_vel = gru_context_vel 
		self.gru_hidden_state_init_vel = gru_hidden_state_init_vel 

		self.gru_context_slane = gru_context_slane 
		self.gru_hidden_state_init_slane = gru_hidden_state_init_slane 

		self.gru_context_lamda = gru_context_lamda 
		self.gru_hidden_state_init_lamda = gru_hidden_state_init_lamda 
				

		# self.gru_context_primal = gru_context_primal 
		# self.gru_hidden_state_init_primal = gru_hidden_state_init_primal


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
		self.A_eq_y = torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0]    ]  )

		self.A_eq = torch.block_diag(self.A_eq_x, self.A_eq_y)
  

				
		# No. of Variables
		self.nvar = P.size(dim = 1)
		self.num = P.size(dim = 0)
		self.num_batch = num_batch
  
		self.A_projection = torch.eye(self.nvar, device = device)
		self.A_y_fin = self.P[-1].reshape(1, self.nvar)

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
		self.rho_lane = 1.0
		self.rho_projection = 1

		self.weight_smoothness = 20.0

		self.P_jerk = torch.diff(self.Pddot, dim = 0)

		# self.cost_smoothness = self.weight_smoothness *( torch.mm(self.Pddot.T, self.Pddot))#+0.1*torch.eye(self.nvar, device = device)
		# self.cost_smoothness = self.weight_smoothness *( torch.mm(self.Pddot.T, self.Pddot)+0.1*torch.eye(self.nvar, device = device))
		# self.cost_smoothness = self.weight_smoothness*(torch.mm(self.P_jerk.T, self.P_jerk))
  
		self.cost_smoothness = self.weight_smoothness *( torch.eye(self.nvar, device = device))
		self.weight_v_des = 2
		self.weight_y_des = 1.0
  
		
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
		self.compute_x_init_batch = torch.vmap(self.compute_x_init, in_dims = (0, 0, 0, 0 )  )
		self.compute_x_batch = torch.vmap(self.compute_x, in_dims = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )  )
		
		


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
		b_eq_y = torch.hstack([y_init_vec, vy_init_vec, ay_init_vec ])
	
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



	def compute_alpha_d(self, primal_sol, x_obs_circle_traj, y_obs_circle_traj, y_ub, y_lb, a_obs, b_obs, lamda_x, lamda_y ):
	 
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
		alpha_v = torch.clip(alpha_v, -0.41*torch.ones(( self.num_batch, self.num  ), device = device), 0.41*torch.ones(( self.num_batch, self.num  ), device = device)   )
		
		c1_d_v = 1.0 * self.rho_ineq * (torch.cos(alpha_v)**2 + torch.sin(alpha_v)**2)
		c2_d_v = 1.0 * self.rho_ineq * (wc_alpha_vx * torch.cos(alpha_v) + ws_alpha_vy * torch.sin(alpha_v))
		
		d_temp_v = c2_d_v/c1_d_v

		d_v = torch.clip(d_temp_v,  torch.tensor(self.v_min).to(device), torch.tensor(self.v_max).to(device))
	
	
		#####################################################################
		
		wc_alpha_ax = xddot
		ws_alpha_ay = yddot
		alpha_a = torch.atan2( ws_alpha_ay+(10e-5), wc_alpha_ax+10e-5)

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
  
	##################################3

		
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
		

		lamda_x = lamda_x - \
	  			  self.rho_ineq * torch.mm(self.A_acc.T, res_ax_vec.T).T - \
			   	  self.rho_ineq * torch.mm(self.A_vel.T, res_vx_vec.T).T - \
				  self.rho_obs * torch.mm(self.A_obs.T, res_x_obs_vec.T).T
				  
		lamda_y = lamda_y - \
	  			  self.rho_ineq * torch.mm(self.A_acc.T, res_ay_vec.T).T - \
			   	  self.rho_ineq * torch.mm(self.A_vel.T, res_vy_vec.T).T - \
				  self.rho_obs * torch.mm(self.A_obs.T, res_y_obs_vec.T).T - \
				  self.rho_lane * torch.mm(self.A_lane.T, res_lane_vec.T).T

		res_norm_batch = torch.linalg.norm(res_acc_vec, dim=1) + \
						 torch.linalg.norm(res_vel_vec, dim=1) +torch.linalg.norm(res_obs_vec, dim = 1)+torch.linalg.norm(res_lane_vec, dim = 1)
		

	   				   
		return alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, alpha_obs, d_obs, s_lane, res_norm_batch
	
	
	
	def compute_x_init(self, b_eq_x, b_eq_y, c_x_guess, c_y_guess):
	 
	 
		b_eq = torch.hstack(( b_eq_x, b_eq_y  ))
		
  
		# A_mat = A_mat_vec.reshape(2*self.nvar, 2*self.nvar)	
  
		cost_smoothness = torch.block_diag(self.A_projection, self.A_projection  )
  
		cost = cost_smoothness

		lincost_x = -self.A_projection.T @ c_x_guess
		lincost_y = -self.A_projection.T @ c_y_guess 
		
		lincost = torch.hstack(( lincost_x, lincost_y   ))

		# lincost = -torch.mm(A_mat.T, b_vec.reshape(2*self.nvar, 1))
  
		# lincost = -A_mat.T @ b_vec
  
	  
		cost_mat = torch.vstack([torch.hstack([cost, self.A_eq.T]), torch.hstack([self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=device)])])
		
		sol = torch.linalg.solve(cost_mat, torch.hstack(( -lincost, b_eq )))
		
		primal_sol = sol[0: 2*self.nvar]

		# primal_sol_x = primal_sol[0:self.nvar]
		# primal_sol_y = primal_sol[self.nvar:2*self.nvar]

		return primal_sol
	
	
	
	def compute_x(self, b_eq_x, b_eq_y, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, lamda_x, lamda_y, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, y_ub, y_lb, s_lane, y_fin):

		
		b_ax_ineq = d_a * torch.cos(alpha_a)
		b_ay_ineq = d_a * torch.sin(alpha_a)

		b_vx_ineq = d_v * torch.cos(alpha_v)
		b_vy_ineq = d_v * torch.sin(alpha_v)
  
		temp_x_obs = d_obs * torch.cos(alpha_obs) * a_obs
		b_obs_x = x_obs_circle_traj + temp_x_obs
		 
		temp_y_obs = d_obs * torch.sin(alpha_obs) * b_obs
		b_obs_y = y_obs_circle_traj + temp_y_obs
		
		# Extending Dimension
		y_ub = y_ub*torch.ones(self.num, device = device)
		y_lb = y_lb*torch.ones(self.num, device = device)
  	
		b_lane = torch.hstack([y_ub * torch.ones(self.num, device=device), -y_lb * torch.ones(self.num, device=device)])
		b_lane_aug = b_lane - s_lane

		y_fin_vec = y_fin*torch.ones(self.num, device = device)
		
  
		# print(v_des.size())
   
		lincost_x = -lamda_x  - \
					self.rho_obs * self.A_obs.T @ b_obs_x - \
		   		 	self.rho_ineq * self.A_acc.T @ b_ax_ineq - \
				 	self.rho_ineq * self.A_vel.T @ b_vx_ineq 
		

		lincost_y = -lamda_y  - \
					self.rho_obs * self.A_obs.T @ b_obs_y - \
		   		 	self.rho_ineq * self.A_acc.T @ b_ay_ineq - \
				 	self.rho_ineq * self.A_vel.T @ b_vy_ineq -\
					self.rho_lane * self.A_lane.T @ b_lane_aug-\
					self.weight_y_des * self.A_y_des.T @ y_fin_vec
			


		cost_x = self.cost_smoothness+self.rho_obs*torch.mm(self.A_obs.T, self.A_obs)+self.rho_ineq*torch.mm(self.A_acc.T, self.A_acc)+self.rho_ineq*torch.mm(self.A_vel.T, self.A_vel)
		cost_y = cost_x+self.rho_lane*torch.mm(self.A_lane.T, self.A_lane)+self.weight_y_des*torch.mm(self.A_y_des.T, self.A_y_des)

			
		cost_mat_x = torch.vstack([torch.hstack([cost_x, self.A_eq_x.T]), torch.hstack([self.A_eq_x, torch.zeros((self.A_eq_x.shape[0], self.A_eq_x.shape[0]), device=device)])])
		cost_mat_y = torch.vstack([torch.hstack([cost_y, self.A_eq_y.T]), torch.hstack([self.A_eq_y, torch.zeros((self.A_eq_y.shape[0], self.A_eq_y.shape[0]), device=device)])])


		sol_x = torch.linalg.solve(cost_mat_x+0.0000*torch.eye( cost_mat_x.size(dim = 1), device = device  ), torch.hstack(( -lincost_x, b_eq_x )))
		sol_y = torch.linalg.solve(cost_mat_y+0.0000*torch.eye( cost_mat_y.size(dim = 1), device = device  ), torch.hstack(( -lincost_y, b_eq_y )))
		

		primal_sol_x = sol_x[0:self.nvar]
		primal_sol_y = sol_y[0:self.nvar]

		primal_sol = torch.hstack([primal_sol_x, primal_sol_y])

		return primal_sol
	
	
	def custom_forward(self, init_state_ego, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, y_ub, y_lb, goal_des, alpha_d_obs, alpha_d_acc, alpha_d_vel, s_lane, lamda, h_0_obs, h_0_a, h_0_v):	

		h_obs = h_0_obs 
		h_a = h_0_a 
		h_v = h_0_v
		
		accumulated_res_primal_list = [] 

		accumulated_res_fixed_point_list= []
		# # Boundary conditions
		b_eq_x, b_eq_y = self.compute_boundary_layer_optim(init_state_ego, goal_des)
		
		alpha_obs = alpha_d_obs[:, 0: self.num_obs*self.num_circles*self.num]
		d_obs = d_obs[:, self.num_obs*self.num_circles*self.num : 2*self.num_obs*self.num_circles*self.num]

		alpha_a = alpha_d_acc[:, 0: self.num ]
		d_a = alpha_d_acc[:, self.num: 2*self.num ]

		alpha_v = alpha_d_vel[:, 0: self.num ]
		d_v = alpha_d_vel[:, self.num: 2*self.num ]

		lamda_x = lamda[:, 0: self.nvar]
		lamda_y = lamda[:, self.nvar: 2*self.nvar]

		
		y_fin = goal_des[:, 1].reshape(self.num_batch, 1)		

		for i in range(0, self.maxiter):

			alpha_obs_prev = alpha_obs.clone()
			d_obs_prev = d_obs.clone() 
			alpha_a_prev = alpha_a.clone()
			d_a_prev = d_a.clone()
			alpha_v_prev = alpha_v.clone()
			d_v_prev = d_v.clone() 
			lamda_x_prev = lamda_x.clone()
			lamda_y_prev = lamda_y.clone()	
			lamda_prev = torch.hstack(( lamda_x_prev, lamda_y_prev  ))
			s_lane_prev = s_lane.clone()

			primal_sol = self.compute_x_batch(b_eq_x, b_eq_y, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, lamda_x, lamda_y, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, y_ub, y_lb, s_lane, y_fin)
	
			alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, alpha_obs, d_obs, s_lane, res_norm_batch = self.compute_alpha_d(primal_sol, x_obs_circle_traj, y_obs_circle_traj, y_ub, y_lb, a_obs, b_obs, lamda_x, lamda_y)

			lamda = torch.hstack(( lamda_x, lamda_y  ))

			r_lamda = torch.hstack(( lamda, lamda_prev, lamda-lamda_prev   ))
			
			r_1_obs = torch.hstack(( alpha_obs_prev, d_obs_prev   ))
			r_2_obs = torch.hstack(( alpha_obs, d_obs   ))

			r_obs = torch.hstack(( r_1_obs, r_2_obs, r_2_obs-r_1_obs  ))


			r_1_a = torch.hstack(( alpha_a_prev, d_a_prev   ))
			r_2_a = torch.hstack(( alpha_a, d_a   ))

			r_a = torch.hstack(( r_1_a, r_2_a, r_2_a-r_1_a  ))
			

			r_1_v = torch.hstack(( alpha_v_prev, d_v_prev   ))
			r_2_v = torch.hstack(( alpha_v, d_v   ))

			r_v = torch.hstack(( r_1_v, r_2_v, r_2_v-r_1_v  ))

			r_s_lane = torch.hstack(( s_lane_prev, s_lane, s_lane-s_lane_prev  ))		

			######################################## obstacle part	
			gru_output_obs, h_obs = self.gru_context(r_obs, h_obs)
			alpha_obs_delta = gru_output_obs[:, 0: self.num_obs*self.num*self.num_circles]
			d_obs_delta = gru_output_obs[:, self.num_obs*self.num*self.num_circles : 2*self.num_obs*self.num*self.num_circles]
			
			alpha_obs = alpha_obs+alpha_obs_delta
			alpha_obs = torch.clip( alpha_obs, -torch.pi*torch.ones(( self.num_batch, self.num_obs*self.num_circles*self.num    ), device = device), torch.pi*torch.ones(( self.num_batch, self.num_obs*self.num_circles*self.num    ), device = device)    )
			d_obs = d_obs+d_obs_delta 
			d_obs = torch.maximum(torch.ones((self.num_batch, self.num * self.num_obs*self.num_circles), device=device), d_obs)


			######################################################################################

			gru_output_a, h_a = self.gru_context_acc(r_a, h_a)
			alpha_a_delta = gru_output_a[:, 0: self.num]
			d_a_delta = gru_output_a[:, self.num : 2*self.num]
			
			alpha_a = alpha_a+alpha_a_delta
			alpha_a = torch.clip( alpha_a, -torch.pi*torch.ones(( self.num_batch, self.num   ), device = device),  torch.pi*torch.ones(( self.num_batch, self.num   ), device = device)   )
			d_a = d_a+d_a_delta 
			d_a = torch.clip(d_a, torch.zeros((self.num_batch, self.num), device=device), torch.tensor(self.a_max).to(device) )
  
			##################################################################################################################


			gru_output_v, h_v = self.gru_context_vel(r_v, h_v)
			alpha_v_delta = gru_output_v[:, 0: self.num]
			d_v_delta = gru_output_v[:, self.num : 2*self.num]
			
			alpha_v = alpha_v+alpha_v_delta
			alpha_v = torch.clip( alpha_v, -torch.pi*torch.ones(( self.num_batch, self.num   ), device = device),  torch.pi*torch.ones(( self.num_batch, self.num   ), device = device)   )
			d_v = d_v+d_v_delta 
			d_v = torch.clip(d_v, torch.zeros((self.num_batch, self.num), device=device), torch.tensor(self.v_max).to(device) )

			###########################################################################################################


			primal_sol_x = primal_sol[:,0: self.nvar]
			primal_sol_y = primal_sol[:,self.nvar:2*self.nvar]

			primal_sol = self.compute_x_init_batch(b_eq_x, b_eq_y, primal_sol_x, primal_sol_y)
			

			accumulated_res_primal_list.append(res_norm_batch)

			fixed_point_residual = torch.linalg.norm(primal_sol-primal_sol_prev, dim = 1)+torch.linalg.norm(lamda_x-lamda_x_prev, dim = 1)+torch.linalg.norm(lamda_y-lamda_y_prev, dim = 1)		
			accumulated_res_fixed_point_list.append(fixed_point_residual)



		
		res_primal_stack = torch.stack(accumulated_res_primal_list )
		res_fixed_point_stack = torch.stack(accumulated_res_fixed_point_list )
		accumulated_res_fixed_point = torch.sum(res_fixed_point_stack, dim = 0)/self.maxiter	
		
  
		accumulated_res_primal = torch.sum(res_primal_stack, dim = 0)/(self.maxiter)
			

		return 	primal_sol, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack
	

	
	def decoder_function(self, inp_norm, init_state_ego, pcd_scaled, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, y_ub, y_lb, param_des, goal_des):
	 
		# PCD feature extractor
		pcd_features = self.point_net(pcd_scaled)

		inp_features = torch.cat([inp_norm, pcd_features], dim = 1)

		alpha_d_obs = self.mlp_init_obs(inp_features)
		alpha_d_acc = self.mlp_init_acc(inp_features)
		alpha_d_vel = self.mlp_init_vel(inp_features)
		s_lane = self.mlp_init_slane(inp_features)
		lamda = self.mlp_init_lamda(inp_features)



		h_0_obs = self.gru_hidden_state_init_obs(inp_features)
		h_0_a = self.gru_hidden_state_init_acc(inp_features)
		h_0_v = self.gru_hidden_state_init_vel(inp_features)
		
		primal_sol, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack =  self.custom_forward(init_state_ego, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, y_ub, y_lb, goal_des, alpha_d_obs, alpha_d_acc, alpha_d_vel, s_lane, lamda, h_0_obs, h_0_a, h_0_v, s_lane, lamda)	
		
		return primal_sol, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack
	


	def ss_loss(self, accumulated_res_primal, primal_sol, accumulated_res_fixed_point):
	 
	
		# primal_loss = (0.5 * (torch.mean(accumulated_res_primal)))

		primal_loss = (0.5 * (torch.mean(accumulated_res_primal)))
		fixed_point_loss = (0.5 * (torch.mean(accumulated_res_fixed_point)))
		jerk_loss = 0.5*torch.mean(torch.linalg.norm(primal_sol, dim =1 ))
  		
		loss = fixed_point_loss#+0.001*jerk_loss
 
		return loss, primal_loss, fixed_point_loss


	def forward(self, inp, init_state_ego, param_des, pcd_data,  closest_obs, psi_obs, dim_x_obs, dim_y_obs,  y_ub, y_lb, P_diag, Pddot_diag, pcd_mean, pcd_std, goal_des):

		# Normalize input
		inp_norm = (inp - self.inp_mean) / self.inp_std
  
		# Batch Trajectory Prediction
		x_obs_circle_traj, y_obs_circle_traj, psi_obs_circle_traj = self.compute_obs_trajectories_batch(closest_obs, psi_obs, dim_x_obs, dim_y_obs)
		a_obs, b_obs = self.compute_obs_ellipse_batch(dim_x_obs, dim_y_obs)
  
		pcd_scaled = (pcd_data - self.min_pcd) / (self.max_pcd - self.min_pcd)

		# pcd_scaled = (pcd_data - pcd_mean) / (self.max_pcd - pcd_std)
				
		# Decode y
		primal_sol, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack = self.decoder_function(inp_norm, init_state_ego, pcd_scaled, x_obs_circle_traj, y_obs_circle_traj, a_obs, b_obs, y_ub, y_lb, param_des, goal_des)
	 
		# predict_traj = (P_diag @ primal_sol.T).T
		# predict_acc = (Pddot_diag @ primal_sol.T).T

		return primal_sol, accumulated_res_primal, accumulated_res_fixed_point
	







		



