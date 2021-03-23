% The input is a deterministic sinusoidal firing rate + a stochastic input
%
% Glosary of terms (Documents vs Code)
%       PDF             MATLAB
%       =================================
%       zeta            r or varsigma
%       xi              x
%       xi_hat          (?)analytic_mean
%       A               F
%       phi(xi)         e_gi or e_ge
%       v0              v0
%       mu              v_e or v_i
%       sigma_1_1       sigma_sq_e
%       sigma_2_2       sigma_sq_i
%       sigma_1_2       cov_e_i
%       sigma_2_1       cov_i_e

function varargout = model_NM_legacy(x,P,mode,params)

% x = states or estimated states
% P = [] or covariance of estimated states

% the parameters
dt = params.dt;

e_0 = params.e0;
r = params.r;	% varsigma
v0 = params.v0; % Threshold

a = params.a; % inverse time constants (excitatory)
b = params.b; % (inhibitory)

A = params.A; % synaptic gains (excitatory)
B = params.B; % (inhibitory)

u = params.mu;	% mean input firing rate.

C = 100;
C1 = 1*C;	% number of synapses
C2 = 0.25*C;

% the states
% Scaling the states (?)
% scale = b*[2/v0, 1, 2/v0, 1]; % Divides by the inhibitory time constant
% x = x .* scale';

v_e = x(1); % Excitatory
z_e = x(2);
v_i = x(3); % Inhibitory
z_i = x(4);

% the state covariance
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

% Linear component of model
F = [1, dt, 0, 0; ...
    -b^2*dt, 1-2*b*dt, 0, 0; ...
    0, 0, 1, dt; ...
    0, 0, -a^2*dt, 1-2*a*dt];

% Sigmoid functions
f_i = 0.5*erf((u - v_i - v0) / (sqrt(2 * (r^2 + sigma_sq_i)))) + 0.5;    % 0.5*erf((u - v_i - v0) / (sqrt(2) * r)) + 0.5;	% inhibitory population firing rate
f_e = 0.5*erf((v_e - v0) / (sqrt(2 * (r^2 + sigma_sq_e)))) + 0.5;        % 0.5*erf((v_e - v0) / (sqrt(2) * r)) + 0.5;        % excitatory population firing rate

alpha_i = B*C2*2*e_0; % lumped constant (inhibitory)
alpha_e = A*C1*2*e_0; % lumped constant (excitatory)

switch mode
    case 'transition'
        % Nonlinear component
        gx = [0; ...
            dt*alpha_e*b*f_i; ...
            0; ...
            dt*alpha_i*a*f_e];
                
        % Nonlinear transition model
        varargout{1} = F*x + gx ;
%         varargout{2} = F*P*F';%P;
        
    case 'analytic'
        % the expectation and covariance of a Gaussian distribution
        % transformed by the NMM        
        % E[Fx + g(x)] = F*E[x] + E[g(x)]
        
        % expectations for inhibitory and excitatory firing rates
        % these erf inputs get re-used (boilerplate)
        input_i = (u - v_i - v0) / (sqrt(2 * (r^2 + sigma_sq_i)));
        input_e = (v_e - v0) / (sqrt(2 * (r^2 + sigma_sq_e)));
        
        e_gi = 0.5*erf(input_i) + 0.5;     % inhibitory population firing rate g(x_i)
        e_ge = 0.5*erf(input_e) + 0.5;     % excitatory population firing rate g(x_e)
        
        % Nonlinear component of expectation
        % E[g(x)] is E_gx, or E[phi(E)]
        E_gx = [0; ...
            dt*alpha_e*b*e_gi; ...
            0; ...
            dt*alpha_i*a*e_ge];
        
        % Expectations for nonlinear components of covariance
        % need to fill out (by expanding the covariance term and using the
        % appropriate solutions)
        
        % f(x) = Fx + g(x)
        
        % P = E[(f(x) - mu)(f(x) - mu)]
        % mu = E[f(x)] = analytic_mean
        % E[(Fx + g(x) - E[Fx + g(x)])(...)]
        %
        
        %         E_x_gx =   % this has a solution E[x_e *  g(x_i)]
        %         E_gx_gx =  % this uses the Gaussian CDF  E[g(x)g(x)]
        %         
        
        % Analytic mean
        analytic_mean = F*x + E_gx;
        
        % Analytic covariance
        %% P1 = APA'
        P1 = F*P*F'; % P2 = Q; % Q is added up in analytic_kalman_filter after calling this function
        
        %% P3 = -E[phi(E)]E[phi(E)'], this is, E[E_phi(E)] = E_gx 
        P3 = -E_gx*E_gx';
        
        %% P4 = -A*E_hat*E[phi(E)'] 
        P4 = -F*x*E_gx'; % Not sure if its F*analytic_mean or F*x
        
        %% P5 = -E[phi(E)](A*E_hat)' 
        P5 = -E_gx*(F*x)';% Not sure if its F*analytic_mean or F*x
        
        %% P6 = E[phi(E)phi(E)'] 
        z_ = v0 + r.*randn(1,2); % Two new independent variables
        z = 0.5*erf((z_ - v0) ./ sqrt(2 * (r^2 + [sigma_sq_e sigma_sq_i].^2))) + 0.5; % Non linear function        
        mu_hat = [v0 - v_e + u, v0 - v_i];               
        
        sigma_hat = [r^2+sigma_sq_e, cov_e_i; ...
                     cov_i_e       , r^2+sigma_sq_i];  
                
        E_gx_gx = mvncdf(z, mu_hat, sigma_hat); % Multivariate Gaussian cumulative distribution 
        
        % Solving MonteCarlo simulation:
        %tic
%         N = 50; % Iterations on the MC
%         % out_ = zeros(size(x,1),size(x,1), N);  % All the values of P (for all the states)
%         out_ = zeros(size(x,1)/2,size(x,1)/2, N); % Only the values of P that correspond to membrane potential
%         for nn = 1:N
%             % MV gaussian
%             % x_ = mvnrnd(x, P, 5000); % All the values of P (for all the states)
%             x_ = mvnrnd(x([1 3],:), P([1 3], [1 3], :), 5000); % Only the values of P that correspond to membrane potential
%             out = non_linear_sigmoid(x_, r, v0); % out = non_linear_sigmoid([v_e z_e v_i z_i], r, v0);
%             out = mean(out);
%             % Matrix multiplication
%             out_(:,:,nn) = out' * out;
%             
%             % Manually compute phi(x)*phi(x)' (considering x is a column vector)
% %             out_(2,2,nn) = out(:,1).*out(:,1);
% %             out_(4,4,nn) = out(:,2).*out(:,2);
% %             out_(2,4,nn) = out(:,1).*out(:,2);
% %             out_(4,2,nn) = out(:,2).*out(:,1);
%         end
%         E_gx_gx = mean(out_,3);
        % P6 = zeros(size(P)); P6([2,4],[2,4]) = E_gx_gx; % Only the values of P that correspond to membrane potential
        %toc
        P6 = E_gx_gx; % dt*E_gx_gx; % All the values of P (for all the states) % Do I need the dt? % This is E_gx_gx or E[g(x)g(x)] equation 25 to 30 SKF_Derivation_Prob
                
        %% P7 and P8 
        % Question: Do this term need to be multiplied by dt?         
        %E_x_gx = [  0    E_xi_gxj(x(1),x(3),v0,P(3,3), P(1,3),r)     0        E_xi_gxj(x(1),x(1),v0,P(1,1), P(1,1),r); ...
        %            0    E_xi_gxj(x(2),x(3),v0,P(3,3), P(2,3),r)     0        E_xi_gxj(x(2),x(1),v0,P(1,1), P(2,1),r); ...
        %            0    E_xi_gxj(x(3),x(3),v0,P(3,3), P(3,3),r)     0        E_xi_gxj(x(3),x(1),v0,P(1,1), P(3,1),r); ...
        %            0    E_xi_gxj(x(4),x(3),v0,P(3,3), P(4,3),r)     0        E_xi_gxj(x(4),x(1),v0,P(1,1), P(4,1),r)];
        
        % E_x_gx = [ 0                     0                          0                           0 ;...
        %           0    E_xi_gxj(x(1),x(1),v0,P(1,1), P(1,1),r)     0        E_xi_gxj(x(1),x(3),v0,P(3,3), P(1,3),r); ...
        %           0                     0                          0                           0 ;...
        %           0    E_xi_gxj(x(3),x(1),v0,P(1,1), P(3,1),r)     0        E_xi_gxj(x(3),x(3),v0,P(3,3), P(3,3),r)];

        E_x_gx = [0                     0                          0                           0 ;...
                  0                     0                          0        E_xi_gxj(x(1),x(3),v0,P(3,3), P(1,3),r); ...
                  0                     0                          0                           0 ;...
                  0    E_xi_gxj(x(3),x(1),v0,P(1,1), P(3,1),r)     0                           0];
        
%         E_x_gx = zeros(size(x,1));
%         for i = 1:size(x,1)
%             for j = 1:size(x,1)
%                 E_x_gx(i,j) = E_xi_gxj(x(i),x(j),v0,P(j,j), P(i,j),r);
%             end
%         end
              
        P7 = E_x_gx;% F*E_x_gx_1'; % This is E_x_gx = E[x_e *  g(x_i)] and E[x_i *  g(x_e)]? equation 22 SKF_Derivation_Prob
                
        P8 = E_x_gx';% E_gx_x;% E_x_gx_2*F'; % This is the transpose of P7? Needs to be confirmed
        
        %% Output:
%         analytic_cov = P1 + 1e-2*(P3 + P4 + P5) + P6 + P7 + P8; % TODO: Fix P3, 4 and 5. Their order of magnitude is too large?
%         analytic_cov = P1 + P3 + P4 + P5 + P6 + P7 + P8; 
        analytic_cov = P1 + 1e-2*(P3 + P4 + P5 + P6 + P7 + P8);
        
        varargout{1} = analytic_mean; % E_t
        varargout{2} = analytic_cov; % P_t
        
    case 'jacobian'        
        % Linearise g()
        f_i_derivative = 2*e_0*r*f_i*(1-f_i);      % inhibitory feedback firing
        f_e_derivative = 2*e_0*r*f_e*(1-f_e);      % excitatory feedback firing
        
        G = [0, 0, 0, 0; ...
            0,0,-dt*B*b*C2*f_i_derivative,0; ...
            0, 0, 0, 0; ...
            dt*A*a*C1*f_e_derivative, 0, 0, 0];
        
        % Jacobian
        varargout{1} = F + G;
end % End switch

end % End function model_NM

function expectation = E_xi_gxj(x_i,x_j, v0, s_jj, s_ij, r)
% Computes the 7th and 8th terms of the P polynomial (P7 and P8). Equation
% 29 from SKF_Derivation_Algebraic:
% E[f(x)] = (x_i/2)*erf((x_j-v0)/sqrt(2(s_jj + r^2))) + x_i/2 + s_ij/sqrt(2*pi*(s_jj + r^2))*exp(-(x_j-v0)^2/(2*(s_jj+r^2)))

expectation = (x_i/2)*erf((x_j-v0)/sqrt(2*(s_jj + r^2))) + x_i/2 +...
    s_ij/sqrt(2*pi*(s_jj + r^2))*exp(-(x_j-v0)^2/(2*(s_jj+r^2)));

end