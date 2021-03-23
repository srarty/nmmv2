% TEST_COVARIANCE_P6 Tests the term P6 of the covariance expectation
% polynomial
%
%   P6 = E[g(x)*g(x)']
%        E_gx_gx
%
% Inputs:
%   N           - Number of iterations for the Monte Carlo
%   N_samples   - (optional) Number of samples in the distribution
%   N_states    - (optional) Number of states. This is the size of the
%                 state vector
%
% Outputs:
%   err - Estimation error. We expect a Gaussian distribution around zero
%   estimate - Output of the Monte Carlo estimation
%   analytic - Output of the analytic solution
%
% Example:
%   err = test_covariance_p6_2(100);
%   histogram(err(1,:));   
%   hold on;
%   histogram(err(2,:));
% 
% Last edit - Artemio Soto, 2020
function [err, varargout] = test_covariance_p6_2(N, varargin)
    N_samples = 1000;
    N_states = 2;
    
    if nargin > 1, N_samples = varargin{1}; end
    if nargin > 2, N_states = varargin{2}; end
    
    
    sample_term_ = zeros(N_states, N_states, N_samples);    % Variable to iterate through the N_samples
    sample_term = zeros(N_states, N_states, N);             % Variable to iterate through the N iterations of the Monte Carlo
    
    mu = [1; 2];        % This vector must have a size: (1,N_states)
    sigma = [0.1 0; 0 0.1]; % This vector must have a size: (1,N_states)
    
    % Check dimmensions of mu and sigma. Throw an error if they don't match
    % with N_states.
    if length(mu)~=N_states || length(sigma)~=N_states
        error('The dimmensions of ''mu'' or ''sigma'' do not correspond to ''N_states''.');
    end
    
    
    r = 3;  % erf sigmoid
    v0 = 6;
    for nn = 1:N
        % MV gaussian
        x = mvnrnd(mu', sigma, N_samples)';
        out = non_linear_sigmoid(x, r, v0, diag(sigma));
        % sample_mean = mean(out,2);
        
        % Matrix multiplication and then take the mean.
        for i = 1:N_samples % Iterate through N_samples
            sample_term_(:,:,i) = (out(:,i)*out(:,i)') .* (x(:,i)*x(:,i)');
        end
        sample_term(:,:,nn) = mean(sample_term_,3); % mean of the N_samples
    end % Expectation from montecarlo
    
    % analytic expectation
    % Bivariate: Multivariate Gaussian cdf neads z to be nXd where n is the number of observations and d is the size of the states (states 'x' is a column vector, in z the states are a row vector)
    z_ = (v0 + r.*randn(1, N_states))'; % 2 new independent random variables z1 and z2. Normal distribution with mean v0 and variance r.
%     z_ = mvnrnd(v0, r, N_states);
    z = non_linear_sigmoid(z_, r, v0);%, diag(sigma));
    mu_hat = [v0 - mu(1); v0 - mu(2)];
    
    sigma_hat = [r^2 + sigma(1,1), sigma(1,2); ...
                 sigma(2,1)    , r^2 + sigma(2,2)];%     sigma_hat = r^2 * eye(N_states) + cov(x(:,1), x(:,2));
    
    E_gx_gx = mvncdf(z', mu_hat', sigma_hat); % Multivariate Gaussian cumulative distribution with mean mu_hat and variance sigma_hat
        
    analytic_term = E_gx_gx .* (mu * mu'); % Analytic expectation
    err = sample_term - analytic_term;
    varargout = {sample_term, analytic_term};
end

