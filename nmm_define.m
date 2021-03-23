% NMM_DEFINE Neural Mass Model based on Jensen and Rit. Defines the state based representation of the model.
%
% Inputs:
%   x - initial states
%   P - initial covariance matrix
%   params - parameters defined by Jensen and Rit
%
% Outputs:
%   nmm - (struct) the neural mass model
%
function nmm = nmm_define(x,P,params)
% Indexes
v_idx = [1 3];
z_idx = [2 4];
u_idx = 5;
alpha_idx = [6 7];

% the parameters
dt          = params.dt;
e_0         = params.e0;
r           = params.r;	% varsigma
v0          = params.v0; % Threshold
decay_e     = params.decay_e; % inverse time constants (excitatory)
decay_i     = params.decay_i; % (inhibitory)
alpha_ei    = params.alpha_ei; % synaptic gains (excitatory)
alpha_ie    = params.alpha_ie; % (inhibitory)
u           = params.u;	% mean input firing rate.

c_constant = 100;
c1 = 1*c_constant;	% number of synapses
c2 = 0.8*c_constant;

% the states
% Scaling the states (?)
% scale = b*[2/v0, 1, 2/v0, 1]; % Divides by the inhibitory time constant
% x = x .* scale';

% Number of augmented states
xlen = length(x);

% Linear component of model
A =     [1,              dt,         0,              0,          0,  0,  0; ...
    -decay_e^2*dt,  1-2*decay_e*dt,  0,              0,          0,  0,  0; ...
         0,              0,          1,              dt,         0,  0,  0; ...
         0,              0    -decay_i^2*dt,   1-2*decay_i*dt,   0,  0,  0; ...
         0,              0,          0,              0,          1,  0,  0; ...
         0,              0,          0,              0,          0,  1,  0; ...
         0,              0,          0,              0,          0,  0,  1];

     
                                                                 
%      
% B =     [0,              0,          0,              0,          0,  0,  0; ...
%          0               0,          0,              0           1,  1,  0; ...
%          0,              0,          0,              0,          0,  0,  0; ...
%          0,              0           0               0           0,  0,  1; ...
%          0,              0,          0,              0,          0,  0,  0; ...
%          0,              0,          0,              0,          0,  0,  0; ...
%          0,              0,          0,              0,          0,  0,  0];
     
% B Matrix (Augmented parameters)
B = zeros(xlen);
% B(z_idx, z_idx) = dt .* diag([decay_e decay_i].*ones(size(z_idx)));
B(z_idx, alpha_idx) = diag(ones(size(z_idx)));

% C Matrix (Augmented)
C = zeros(xlen);
C(2,3) = 1; % inhibitory -> excitatory
C(4,1) = 1; % excitatory -> inhibitory
C(2,u_idx) = 1; % input -> excitatory

alpha_i = alpha_ei * c1 * 2 * e_0 * dt * decay_e; % lumped constant (inhibitory, input to)
alpha_e = alpha_ie * c2 * 2 * e_0 * dt * decay_i; % lumped constant (excitatory, input to)

x(6) = alpha_e;
x(7) = alpha_i;

nmm = struct;
nmm.A = A;
nmm.B = B;
nmm.C = C;
nmm.x0 = x;
nmm.P0 = P;
nmm.params = params;
% nmm.t = 1;

end % end function - nmm_define
