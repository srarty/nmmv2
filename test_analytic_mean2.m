% TEST_ANALYTIC_MEAN2 Tests the mean of a gaussian transformed by the erf
% sigmoid function.
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
%   err = test_analytic_mean2(100);
%   histogram(err);
%
% Created - Pip Karoly, 2020
% Last edit - Artemio Soto, 2020
function err = test_analytic_mean2(N, varargin)
    % TESTING THE MEAN OF A GAUSSIAN TRANSFORMED BY ERF SIGMOID
    N_samples = 1000;
    N_states = 2;
    
    if nargin > 1, N_samples = varargin{1}; end
    if nargin > 2, N_states = varargin{2}; end
    
    sample_mean = nan(N_states,N);
    
    mu = [2 3]; % This vector must have a size: (1,N_states)
    sigma = [0.5 1.5]; % This vector must hace a size: (1,N_states)
    
    % Check dimmensions of mu and sigma. Throw an error if they don't match
    % with N_states.
    if length(mu)~=N_states || length(sigma)~=N_states
        error('The dimmensions of ''mu'' or ''sigma'' do not correspond to ''N_states''.');
    end
        
    r = 3;  % erf sigmoid
    v0 = 6;
    % MV gaussian
    mu_ = mu;
    sigma_ = sigma;
    for nn = 1:N
        x = mvnrnd(mu_, sigma_, N_samples);
        out = non_linear_sigmoid(x, r, v0);
        sample_mean(:,nn) = mean(out);
    end

    % analytic mean
    input_i = (mu(1) - v0) / (sqrt(2 * (r^2 + sigma(1))));
    input_e =  (mu(2) - v0) / (sqrt(2 * (r^2 + sigma(2))));
    analytic_mean = 0.5*erf([input_i input_e]) + 0.5; % excitatory population firing rate E[g(x_e)]
    err = sample_mean - analytic_mean';
end

function out = non_linear_sigmoid(x, r, v0)
    input = (x - v0) / (sqrt(2) * r);
    out = 0.5*erf(input) + 0.5; % excitatory population firing rate
end
