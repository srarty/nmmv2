% TEST_COVARIANCE_P3 Tests the term P3 of the covariance expectation
% polynomial
%
%   P3 = -E_gx*E_gx'
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
%   err = test_covariance_p3(500);
%   histogram(err);
%
% Last edit - Artemio Soto, 2020
function err = test_covariance_p3(N, varargin)
    N_samples = 1000;
    N_states = 2;
    
    if nargin > 1, N_samples = varargin{1}; end
    if nargin > 2, N_states = varargin{2}; end
    
    sample_mean = nan(N_states, N);
    sample_term = zeros(N_states, N_states, N);
    
    mu = 5*rand(1,N_states);%[-1 2]; % This vector must have a size: (1,N_states)
    sigma = rand(N_states);%[0.4 , 0; ...
             %0 , 0.1]; % This matrix must have a size: (N_states,N_states)
    
    % Check dimmensions of mu and sigma. Throw an error if they don't match
    % with N_states.
    if length(mu)~=N_states || length(sigma)~=N_states
        error('The dimmensions of ''mu'' or ''sigma'' do not correspond to ''N_states''.');
    end
    
    
    r = 3;  % erf sigmoid
    v0 = 6;
    % MV gaussian
    mu_ = mu;
    sigma_ = diag(sigma)';
    for nn = 1:N
        x = mvnrnd(mu_, sigma_, N_samples);
        out = non_linear_sigmoid(x, r, v0);
        sample_mean(:,nn) = mean(out);
        sample_term(:,:,nn) = sample_mean(:,nn)*sample_mean(:,nn)';
    end

    % analytic expectation
    input_i = (mu(1) - v0) / (sqrt(2 * (r^2 + sigma(1,1))));
    input_e =  (mu(2) - v0) / (sqrt(2 * (r^2 + sigma(2,2))));
    E_gx = transpose(0.5*erf([input_i input_e]) + 0.5); % excitatory population firing rate E[g(x_e)]
    analytic_term = E_gx * E_gx'; % Analytic expectation
    err = sample_term - analytic_term;
end

function out = non_linear_sigmoid(x, r, v0)
    input = (x - v0) / (sqrt(2) * r);
    out = 0.5*erf(input) + 0.5;
end
