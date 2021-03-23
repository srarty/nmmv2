% TEST_COVARIANCE_P7 Tests the term P7 of the covariance expectation
% polynomial
%
%   P7 = E[Fx * g(x)']
%        E_x_gx
%
% Inputs:
%   N           - Number of iterations for the Monte Carlo
%   N_samples   - (optional) Number of samples in the distribution
%   N_states    - (optional) Number of states. This is the size of the
%                 state vector
%
% Outputs:
%   err - Estimation error. We expect a Gaussian distribution around zero
%
% Example:
%   err = test_covariance_p7(500);
%   histogram(err(1,:)); 
%   hold on
%   histogram(err(2,:)); 
%
% Last edit - Artemio Soto, 2020
function err = test_covariance_p7(N, varargin)
    N_samples = 1000;
    N_states = 2;
    
    if nargin > 1, N_samples = varargin{1}; end
    if nargin > 2, N_states = varargin{2}; end
    
    sample_term = zeros(N_states, N_states, N);
    sample_term_ = zeros(N_states, N_states, N_samples);
    
    mu = 10*rand(1,N_states);   %[2 0.5]; % This vector must have a size: (1,N_states)
    sigma = rand(N_states)^rand(1); %[0.1 , 0; ...
                            % 0   , 0.2]; % This vector must have a size: (1,N_states)
% 	sigma = (sigma + sigma')/2;% Make sigma symmetric
    sigma = nearestSPD(sigma); % Make sigma positive definite (the result might be positive semi-definite)
    
    % Check dimmensions of mu and sigma. Throw an error if they don't match
    % with N_states.
    if length(mu)~=N_states || length(sigma)~=N_states
        error('The dimmensions of ''mu'' or ''sigma'' do not correspond to ''N_states''.');
    end
    
    r = 3;  % erf sigmoid
    v0 = 6;
    % MV gaussian
    mu_ = mu;
    for nn = 1:N
        x = mvnrnd(mu_, sigma, N_samples);
        out = non_linear_sigmoid(x, r, v0);
        % Manually compute the product
%         sample_term(1,nn) = mean(x(:,2).*out(:,1)); % <one>
%         sample_term(2,nn) = mean(x(:,1).*out(:,2)); % <two>
%         sample_term(3,nn) = mean(x(:,1).*out(:,1)); % <three>
%         sample_term(4,nn) = mean(x(:,2).*out(:,2)); % <four>
        % Multiply both vectors
        for i = 1:N_samples
            sample_term_(:,:,i) = x(i,:)'.*out(i,:);
        end
        sample_term(:,:,nn) = mean(sample_term_, 3);
    end
    
    % analytic expectation
    input_i = (mu(1) - v0) / (sqrt(2 * (r^2 + sigma(1,1))));
    input_e =  (mu(2) - v0) / (sqrt(2 * (r^2 + sigma(2,2))));
    
    e_gi = 0.5*erf(input_i) + 0.5; % excitatory population firing rate E[g(x_e)]
    e_ge = 0.5*erf(input_e) + 0.5; % excitatory population firing rate E[g(x_e)]
    
    E_xe_gxi = mu(2) * e_gi + sigma(2,1)/(sqrt(2*pi*(sigma(1,1) + r^2)))*exp(-input_i^2); % <one>
    E_xi_gxe = mu(1) * e_ge + sigma(1,2)/(sqrt(2*pi*(sigma(2,2) + r^2)))*exp(-input_e^2); % <two>
    E_xe_gxe = mu(1) * e_gi + sigma(1,1)/(sqrt(2*pi*(sigma(1,1) + r^2)))*exp(-input_i^2); % <three>
    E_xi_gxi = mu(2) * e_ge + sigma(2,2)/(sqrt(2*pi*(sigma(2,2) + r^2)))*exp(-input_e^2); % <four>

    analytic_term = [E_xe_gxe E_xe_gxi; E_xi_gxe E_xi_gxi ]; % Analytic expectation
    err = sample_term - analytic_term';
end

function out = non_linear_sigmoid(x, r, v0)
    input = (x - v0) / (sqrt(2) * r);
    out = 0.5*erf(input) + 0.5;
end
