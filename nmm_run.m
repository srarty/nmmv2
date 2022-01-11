% NMM_RUN Runs one iteration of the neural mass model (propagates mean and covariance at time t)
% 
%   Inputs: 
%       nmm - (struct) A predefined neural mass model.
%
%   Outputs:
%       x - States vector at t + 1
%       P - Covariance matrix at t + 1
%       f_e - Output of nonlinearity (excitatory)
%       f_i - Output of nonlinearity (inhibitory)
%
% The University of Melbourne | Artemio - March 2021
%
function varargout = nmm_run(nmm, x, P, mode)
% Model
A       = nmm.A;
B       = nmm.B;
C       = nmm.C;

% Indexes
v_idx       = [1 3];
z_idx       = [2 4];
u_idx       = 5;
alpha_idx   = [6 7];

NStates = length([v_idx z_idx u_idx alpha_idx]);
NSynapses = length(v_idx);

% Options]
MONTECARLO = false; % (Default) montecarlo disabled, P6 is calculated analytically, unless nmm.options.P6_montecarlo exists and is set as true
if isfield(nmm.options, 'P6_montecarlo'), MONTECARLO = nmm.options.P6_montecarlo; end % P6 computation (montecarlo or analytical)

ALPHA_DECAY = false; 
if isfield(nmm.options, 'ALPHA_DECAY'), ALPHA_DECAY = nmm.options.ALPHA_DECAY; end 


% The parameters
dt          = nmm.params.dt;
e_0         = nmm.params.e0;
r           = nmm.params.r;         % varsigma
v0          = nmm.params.v0;        % Threshold
decay_e     = nmm.params.decay_e;   % inverse time constants (excitatory)
decay_i     = nmm.params.decay_i;   % (inhibitory)
alpha_ei    = nmm.params.alpha_ei;  % synaptic gains (excitatory)
alpha_ie    = nmm.params.alpha_ie;  % (inhibitory)
u           = nmm.params.u;         % mean input firing rate.

% Implement exponential decay on alpha - params
if ALPHA_DECAY
    x(alpha_idx(1)) = x(alpha_idx(1)) * (0.99^(dt*decay_e/1)); % alpha_e; % Exponential decay implemented
    x(alpha_idx(2)) = x(alpha_idx(2)) * (0.99^(dt*decay_e/1)); %alpha_i; % Exponential decay implemented
end

% The (augmented) states vector
v_e     = x(1); % Excitatory
z_e     = x(2); % Derivative (excitatory)
v_i     = x(3); % Inhibitory
z_i     = x(4); % Derivative (inhibitory)
u       = x(5); % External input
a_ei    = x(6); % Synaptic strength (e to i)
a_ie    = x(7); % Synaptic strength (i to e)

% The state covariance
sigma_vector = zeros(size(x));
if ~isempty(P)
    sigma_sq_e = P(1,1);  % excitatory
    sigma_sq_i = P(3,3);  % inhib
    
    cov_e_i = P(1,3);
    cov_i_e = P(3,1);
    
    sigma_vector(2) = sigma_sq_i; 
    sigma_vector(4) = sigma_sq_e;   
else
    sigma_sq_e = 0;  % excitatory
    sigma_sq_i = 0;  % inhib
    
    cov_e_i = 0;
    cov_i_e = 0;
end

% The numeric solution of P6
w_fast = [0.1713244923791705 0.3607615730481384  0.4679139345726904 0.1713244923791705];%  0.3607615730481384 0.4679139345726904];
y_fast = [0.033765242898424 0.169395306766868 0.380690406958402 0.966234757101576];% 0.830604693233132 0.619309593041599];

% Sigmoid functions
% f_e = non_linear_sigmoid(u - v_i, r, v0, sigma_sq_i); % 0.5*erf((u - v_i - v0) / (sqrt(2 * (r^2 + sigma_sq_i)))) + 0.5;    % 0.5*erf((u - v_i - v0) / (sqrt(2) * r)) + 0.5;	   % excitatory population firing rate
% f_i = non_linear_sigmoid(  v_e,   r, v0, sigma_sq_e); % 0.5*erf((v_e - v0) / (sqrt(2 * (r^2 + sigma_sq_e)))) + 0.5;        % 0.5*erf((v_e - v0) / (sqrt(2) * r)) + 0.5;        % inhibitory population firing rate
% 
% phi = [f_e; f_i]; % Only the important indexes. This phi should be equal to the 2nd and 4th entries of phi_c

C_inhibit = C; % Auxiliary matrix C_inhibit, same as C but the inhibitory element is negative to include it in the nonlinearity as inhibition
C_inhibit(2,3) = -C_inhibit(2,3);
phi_c = non_linear_sigmoid(C_inhibit*x,r,v0, sigma_vector); % Complete phi from C
switch mode
    case 'transition'                
        % Nonlinear transition model
        varargout{1} = A*x + B*x.*phi_c; % Mean propagation. Forward model
%         varargout{2} = A*P*A';
        
    case 'analytic'
        % the expectation and covariance of a Gaussian distribution
        % transformed by the NMM        
        % E[Fx + g(x)] = F*E[x] + E[g(x)]
        
        % expectations for inhibitory and excitatory firing rates
        % these erf inputs get re-used (boilerplate)
        %{
        input_i = (u - v_i - v0) / (sqrt(2 * (r^2 + sigma_sq_i)));
        input_e = (v_e - v0) / (sqrt(2 * (r^2 + sigma_sq_e)));
                
        e_gi = 0.5*erf(input_i) + 0.5;     % inhibitory population firing rate g(x_i)
        e_ge = 0.5*erf(input_e) + 0.5;     % excitatory population firing rate g(x_e)
        %}
        
        % Nonlinear component of expectation
        % E[g(x)] is E_gx, or E[phi(X)]
        % E_gx = B*x .* phi_c; % non_linear_sigmoid(C*x,r,v0)        
%         phi = B*x .* phi_c; % E_gx = phi_c; % non_linear_sigmoid(C*x,r,v0) % This is phi(Xi)
        phi_b = B*x;        
        E_gx = zeros(size(x));
        E_gx(2) = E_phi(x(2), -x(3)+x(5), v0, P(3,3), P(2,3), r); % E_phi(phi_b(2),phi_c(2),v0,P(1,1), P(1,3),r);        
        E_gx(4) = E_phi(x(4), x(1), v0, P(1,1), P(4,1), r);% E_phi(phi_b(4),phi_c(4),v0,P(3,3), P(3,1),r);        
        
%         E_gx = zeros(size(phi,1)); % Full matrix
%         phi_ = phi(z_idx);
%         E_gx_ = zeros(size(phi_,1)); % 2 x 2 matrix
%         for i = 1:size(phi_,1)
%             for j = 1:size(phi_,1)
%                 E_gx_(i,j) = E_phi(phi_(i),phi_(j),v0,P(j,j), P(i,j),r);
%             end
%         end
%         E_gx(z_idx,z_idx) = E_gx_; % This is E_gx = E[phi(Xi)]


        CPC = C(z_idx,:)*P*C(z_idx,:)';
        dCPB = diag(C(z_idx,:)*P*B(z_idx,:)');
        Bxi = B(z_idx,alpha_idx)*x(alpha_idx);
        
        gamma = 1./sqrt(2*(diag(CPC) + r^2));
        beta = (C_inhibit(z_idx,[v_idx u_idx])*x([v_idx u_idx]) - v0).*gamma;       % <- v_0
        Xi = (erf(beta) + 1)/2;
%         Xi = non_linear_sigmoid(beta,r,v0);
        Upsilon = exp(-(beta.^2)).*gamma./sqrt(pi);
        psi = Bxi.*Xi + dCPB.*Upsilon;                              % E[ Bxi_t o g(Cxi_t) ]
        
        %% Analytic mean
        analytic_mean = A*x;
        analytic_mean(2:2:2*NSynapses) = analytic_mean(2:2:2*NSynapses) + psi;
        
        % analytic_mean = A*x + B*x.*phi; 
        
        %% Analytic covariance (Artemio)
        %% P1 = APA'
        P1 = A*P*A'; % P2 = Q; % Q is added in analytic_kalman_filter_2 after calling nmm function
        
        %% P3 = -E[phi(E)]E[phi(E)'], this is, E[E_phi(E)] = E_gx 
        P3 = -(E_gx * E_gx');
        
        %% P4 = -A*Xi_hat*E[phi(Xi)'] 
        P4 = -A*x * E_gx'; % -A*x*E_gx';
        
        %% P5 = -E[phi(E)](A*E_hat)' 
        P5 = -E_gx * (A*x)';% -E_gx*(A*x)';
        
        %% P6 = E[phi(E)phi(E)'] 
        P6_ =  E_gx_gx(x, P, u, v0, r, MONTECARLO); % All the values of P (for all the states) % Do I need the dt? % This is E_gx_gx or E[g(x)g(x)] equation 25 to 30 SKF_Derivation_Prob
        P6 = zeros(NStates);
        P6(z_idx,z_idx) = P6_;
        
        %% P7 and P8
        %P7 = E_x_gx;% F*E_x_gx_1'; % This is E_x_gx = E[x_e *  g(x_i)] and E[x_i *  g(x_e)]? equation 22 SKF_Derivation_Prob
        % E_xi_gxj(u1, u2, u3, v0, s12, s13, s23, s33, r)
        P7 = zeros(NStates, NStates);
%         P7(2,2) = E_xi_gxj(x(2), x(2), phi_c(2), v0, P(2,2), P(2,3), P(2,3), P(3,3), r);
%         P7(2,4) = E_xi_gxj(x(2), x(4), phi_c(2), v0, P(2,4), P(2,3), P(4,3), P(3,3), r);
%         P7(4,2) = E_xi_gxj(x(4), x(2), phi_c(2), v0, P(4,2), P(4,1), P(2,1), P(1,1), r);
%         P7(4,4) = E_xi_gxj(x(4), x(4), phi_c(2), v0, P(4,4), P(4,1), P(4,1), P(1,1), r);
        P7(2,2) = E_xi_gxj(x(2), x(2), -x(3)+x(5), v0, P(2,2), P(2,3), P(2,3), P(3,3), r);
        P7(2,4) = E_xi_gxj(x(2), x(4), -x(3)+x(5), v0, P(2,4), P(2,3), P(4,3), P(3,3), r);
        P7(4,2) = E_xi_gxj(x(4), x(2), x(1), v0, P(4,2), P(4,1), P(2,1), P(1,1), r);
        P7(4,4) = E_xi_gxj(x(4), x(4), x(1), v0, P(4,4), P(4,1), P(4,1), P(1,1), r);
                

        P8 = P7';%E_x_gx';% E_gx_x;% E_x_gx_2*F'; % This is the transpose of P7
                        
        % P4, P5, P7 and P8 are covered by the following lines
        % Missing covariance terms (updates alpha params):
        q2 = Upsilon.*(Bxi - dCPB.*beta.*gamma.^2 *2);
        AP = A*P;
        Phi = ones(NStates,1)*q2'.*(AP*C_inhibit(z_idx,:)') + ones(NStates,1)*Xi'.*(AP*B(z_idx,:)');
                
        %% Output:
        % Uncomment next line for full polynomial as above.
%         analytic_cov = P1 + P3 + P6 + P4 + P5 + P7 + P8; % Full polynomial
        % Comment previous line and uncomment following three to calculate
        % P4, P5, P7 and P8 as Phi + Phi'
        analytic_cov = P1 + P3 + P6;% + P4 + P5 + P7 + P8;
        analytic_cov(:,z_idx) = analytic_cov(:,z_idx) + Phi; % This is P4 and P7
        analytic_cov(z_idx,:) = analytic_cov(z_idx,:) + Phi'; % This is P5 and P8
        
        varargout{1} = analytic_mean; % E_t
        varargout{2} = analytic_cov; % P_t
    
    case 'pip'
        CPC = C(z_idx,:)*P*C(z_idx,:)';
        dCPB = diag(C(z_idx,:)*P*B(z_idx,:)');
        Bxi = B(z_idx,alpha_idx)*x(alpha_idx);
        
        gamma = 1./sqrt(2*(diag(CPC) + r^2));
        beta = (C_inhibit(z_idx,[v_idx u_idx])*x([v_idx u_idx]) - v0).*gamma;       % <- v_0
        Xi = (erf(beta) + 1)/2;
        Upsilon = exp(-(beta.^2)).*gamma./sqrt(pi);
        psi = Bxi.*Xi + dCPB.*Upsilon;                              % E[ Bxi_t o g(Cxi_t) ]
        
        %% Analytic mean
        analytic_mean = A*x;
        analytic_mean(2:2:2*NSynapses) = analytic_mean(2:2:2*NSynapses) + psi;
        
        %% Analytic Covariance (Pip)
        % cov part 1 (Phi)
        q2 = Upsilon.*(Bxi - dCPB.*beta.*gamma.^2 *2);
        AP = A*P;        
        Phi = ones(NStates,1)*q2'.*(AP*C_inhibit(z_idx,:)') + ones(NStates,1)*Xi'.*(AP*B(z_idx,:)');

        % note: A term A*xi_t * E[ Bxi_t o g(Cxi_t) ]' = A*xi_0p*psi' is cancelled
        % from here from the full covariance expansion.

        % cov part 2 (Omega)
        %
        % NOTE: we can reduce the dimensionality (only compute upper triangle part) of the matrices we turn to vectors to increase speed in larger networks.
        %
        CPCgammagamma = asin(CPC.*(gamma*gamma')*2);
        CPCgammagamma = CPCgammagamma(:);                                           % change to a vector
        CPCgammagamma_y = CPCgammagamma*y_fast;                                     % a matrix of row vectors

        betabeta = (beta*beta')*2;
        betabeta_mat = betabeta(:)*ones(1,length(w_fast));                          % change to a vector and repmat

        beta2mat = (beta.^2)*ones(1,NSynapses);                                         % sq and repmat
        beta2matT = beta2mat';                                                      % the transpose allow for permutations when we sum below
        beta2_plus_beta2T = (beta2mat(:) + beta2matT(:))*ones(1,length(w_fast));    % change to a vectors, add, and repmat

        % put it together
        %
        Psi = reshape(sum(CPCgammagamma*w_fast.*exp(-(beta2_plus_beta2T - betabeta_mat.*sin(CPCgammagamma_y))./cos(CPCgammagamma_y).^2),2),NSynapses,NSynapses)/(4*pi);
        %                 ~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %             ~~~
        Omega = ((Xi*Xi') + Psi).*(Bxi*Bxi' + P(alpha_idx,alpha_idx));
        %        ~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        %         E[g^2(Cxi)]           (alpha^2 + sigma_alpha^2)


        %%
        % here we construct the cov mat.
        %
        P_1m = AP*A';% + Q; % Q added outside this function
        P_1m(z_idx,z_idx) = P_1m(z_idx,z_idx)  + Omega - psi*psi';
        P_1m(:,z_idx) = P_1m(:,z_idx) + Phi;
        P_1m(z_idx,:) = P_1m(z_idx,:) + Phi';
        analytic_cov = P_1m;
        
        
        varargout{1} = analytic_mean; % E_t
        varargout{2} = analytic_cov; % P_t

    case 'jacobian'        
        % Linearise g()
%         f_i_derivative = 2*e_0*r*f_i*(1-f_i);      % inhibitory feedback firing
%         f_e_derivative = 2*e_0*r*f_e*(1-f_e);      % excitatory feedback firing
%         
%         G = [   0,              0,              0,              0,	0,  0,  0; ...
%                 0,              0,     -a_ie*f_i_derivative,    0,	0,  0,  0; ...
%                 0,              0,              0,              0,	0,  0,  0; ...
%         a_ei*f_e_derivative,    0,              0,              0,	0,  0,  0; ...
%                 0,              0,              0,              0,  0,  0,  0; ...
%                 0,              0,              0,              0,  0,  0,  0; ...
%                 0,              0,              0,              0,  0,  0,  0];
%         
%         % Jacobian
%         varargout{1} = A + G;

        % Derivatives of non linear function:
        df_e = r*f_e*(1-f_e); %0.5/(r*sqrt(2));
        df_i = r*f_i*(1-f_i); % 0.5/(r*sqrt(2));
        
        % Jacobian Matrix
        J = [   1,             dt,              0,              0,          0,          0,      0; ...
         -decay_e^2*dt,   1-2*decay_e*dt,   -x(6)*df_e,      	0,      x(6)*df_e,     f_e      0; ...
                0,              0,              1,             dt,          0,          0,      0; ...
            x(7)*df_i,          0       -decay_i^2*dt,   1-2*decay_i*dt,    0,          0,     f_i; ...
                0,              0,              0,              0,          1,          0,      0; ...
                0,              0,              0,              0,          0,          1,      0; ...
                0,              0,              0,              0,          0,          0,      1];
        
        varargout{1} = J;
            
end % End switch

% Optional outputs
varargout{3} = phi_c(2);%f_e;
varargout{4} = phi_c(4);%f_i;

% Increase the counter of iterations if everything went well.
% nmm.t = nmm.t + 1; 

end % End function

%% Function to calculate P6 (equation 14)
function expectation = E_gx_gx(x, P, u, v0, r, montecarlo, varargin)
    % Computes the 6th term of the analytic covariance polynomial
    % Inputs:
    %   montecarlo - If 'true', it calculates it in a montecarlo
    %   N (varargin) - number of montecarlo iterations
    % State covariance
    if ~isempty(P)
        sigma_sq_e = P(1,1);  % excitatory
        sigma_sq_i = P(3,3);  % inhib
        cov_e_i = P(1,3);
        cov_i_e = P(3,1);
    else
        sigma_sq_e = 0;  % excitatory
        sigma_sq_i = 0;  % inhib
        cov_e_i = 0;
        cov_i_e = 0;
    end
    
    if montecarlo
        N = 50;
        N_samples = 1;
        
        if nargin > 6, N = varargin{1}; end
        if nargin > 7, N_samples = varargin{2}; end
        
        % Solve statistically
        %tic
        sample_term_ = zeros(2, 2, N_samples);    % Variable to iterate through the N_samples
        sample_term = zeros(2, 2, N);             % Variable to iterate through the N iterations of the Monte Carlo
        
        sigma = [sigma_sq_e,    cov_e_i     ;...
                 cov_i_e,       sigma_sq_i  ];
             
        for nn = 1:N
            % MV gaussian
            x_ = mvnrnd([x(1) x(3)], sigma, N_samples)';
            out = non_linear_sigmoid(x_, r, v0, diag(sigma));
            % sample_mean = mean(out,2);
            
            % Matrix multiplication and then take the mean.
            for i = 1:N_samples % Iterate through N_samples
                sample_term_(:,:,i) = (out(:,i)*out(:,i)') .* (x_(:,i)*x_(:,i)');
            end
            sample_term(:,:,nn) = mean(sample_term_,3); % mean of the N_samplesh
        end % Expectation from montecarlo
        expectation = mean(sample_term,3);  % Statistical expectation
        %toc
    else
        % If montecarlo is 'false', calculate the expectation analytically
        
        % States (membrane potentials)
        v_e = x(1); % Excitatory
        v_i = x(3); % Inhibitory
                
        % Solve analytically
        % analytic expectation
        % Bivariate: Multivariate Gaussian cdf neads z to be nXd where n is the number of observations and d is the size of the states (states 'x' is a column vector, in z the states are a row vector)
        %z_ = (v0 + r.*randn(1, 2))'; % 2 new independent random variables z1 and z2. Normal distribution with mean v0 and variance r. % z_ = mvnrnd(v0, r, N_states);
        %z = non_linear_sigmoid(z_, r, v0, [sigma_sq_e sigma_sq_i]);
        mu_hat = [v0 - (x(5) - v_i); v0 - v_e];
        
        sigma_hat = [r^2 + sigma_sq_e, cov_e_i; ...
                     cov_i_e    , r^2 + sigma_sq_i];%     sigma_hat = r^2 * eye(N_states) + cov(x(:,1), x(:,2));
        
        % E_gx_gx = mvncdf(z', mu_hat', sigma_hat); % Multivariate Gaussian cumulative distribution with mean mu_hat and variance sigma_hat
        E_gx = mvncdf([0 0; 0 0], mu_hat', sigma_hat); % Multivariate Gaussian cumulative distribution with mean mu_hat and variance sigma_hat. This corresponds to 1 Phi function
        E_gx_gx = E_gx; % This corresponds to Phi*Phi'
    
        expectation = E_gx_gx .* ([x(6);x(7)] * [x(6);x(7)]'); % Analytic expectation
    end
end

%% Function to E[phi(Xi)] (equation 12)
function expectation = E_phi(x_i,x_j, v0, s_jj, s_ij, r)
% Computes Equation 29 from SKF_Derivation_Algebraic:
% E[f(x)] = (x_i/2)*erf((x_j-v0)/sqrt(2(s_jj + r^2))) + x_i/2 + s_ij/sqrt(2*pi*(s_jj + r^2))*exp(-(x_j-v0)^2/(2*(s_jj+r^2)))

	expectation = (x_i/2)*erf((x_j-v0)/sqrt(2*(s_jj + r^2))) + x_i/2 +...
        (s_ij/sqrt(2*pi*(s_jj + r^2)))*exp(-(x_j-v0)^2/(2*(s_jj+r^2)));

end

%% Function to E[Xi*phi(Xi)] (equation 13)
function expectation = E_xi_gxj(u1, u2, u3, v0, s12, s13, s23, s33, r)
% Computes Equation 37 from SKF_Derivation_Algebraic:
	expectation = ((u1*u2+s12)/2)*(1+erf((u3-v0)/sqrt(2*(r^2 + s33)))) -...
        ((s13*s23*(u3-v0)/(sqrt(2*pi)*(r^2 + s33)^(3/2))) - ((s23*u1+s13*u2)/sqrt(2*pi*(r^2 + s33)))) *...
        exp(-(u3-v0)^2 /(2*(r^2 + s33)));
end
