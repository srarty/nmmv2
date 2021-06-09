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
%    err = test_covariance_p6_2(500);
%    histogram(err(1,1,:));
%    hold on
%    histogram(err(1,2,:));
%    histogram(err(2,2,:));
% 
% Last edit - Artemio Soto, 2020
function [err, varargout] = test_covariance_p6_2(N, varargin)
    N_samples = 1;
    N_states = 2;
    
    if nargin == 0, N = 1000; end
    if nargin > 1, N_samples = varargin{1}; end
    if nargin > 2, N_states = varargin{2}; end
    
    
    sample_term_ = zeros(N_states, N_states, N_samples, N);    % Variable to iterate through the N_samples
    sample_term = zeros(N_states, N_states, N);             % Variable to iterate through the N iterations of the Monte Carlo
    
    mu = [1; 10];        % This vector must have a size: (1,N_states)
    sigma = [10 0; 0 3]; % This matrix must have a size: (N_states,N_states)
    
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
        phi = non_linear_sigmoid(x, r, v0, diag(sigma));
        % sample_mean = mean(out,2);
        
        % Matrix multiplication and then take the mean.
        for i = 1:N_samples % Iterate through N_samples
            sample_term_(:,:,i,nn) = (phi(:,i)*phi(:,i)') .* ([1; 1] * [1 1]);%(x(:,i)*x(:,i)');
        end
    end % Expectation from montecarlo
    sample_term(:,:,:) = mean(sample_term_,3); % mean of the N_samples
    
    % analytic expectation
    % Bivariate: Multivariate Gaussian cdf neads z to be nXd where n is the number of observations and d is the size of the states (states 'x' is a column vector, in z the states are a row vector)
    z_ = (v0 + r.*randn(1, N_states))'; % 2 new independent random variables z1 and z2. Normal distribution with mean v0 and variance r.
%     z_ = mvnrnd(v0, r, N_states);
    z = non_linear_sigmoid(z_, r, v0);%, diag(sigma));
    mu_hat = [v0 - mu(1); v0 - mu(2)];
    
    sigma_hat = [r^2 + sigma(1,1), sigma(1,2); ...
                 sigma(2,1)    , r^2 + sigma(2,2)];%     sigma_hat = r^2 * eye(N_states) + cov(x(:,1), x(:,2));
    
    E_gx = mvncdf([0 0; 0 0], mu_hat', sigma_hat); % Multivariate Gaussian cumulative distribution with mean mu_hat and variance sigma_hat. This corresponds to 1 Phi function
    E_gx_gx = E_gx * E_gx'; % This corresponds to Phi*Phi'
        
    analytic_term = E_gx_gx .* ([1; 1] * [1 1]); % Analytic expectation
    err = sample_term - analytic_term;
    varargout = {sample_term, analytic_term};
    
    figure
    histogram(err(1,1,:));
    hold on
    histogram(err(1,2,:));
    histogram(err(2,2,:));
end

