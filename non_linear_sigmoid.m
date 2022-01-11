% NON_LINEAR_SIGMOID Computes the non linear function for the Neural Mass 
% Model.
%
%   Implements: 0.5 * erf[(x - v0)/sqrt(2*(r^2 + sigma^2))] + 0.5
%
% Inputs:
%   x       -   The states vector. The rows of x are the states, the
%               columns are the samples
%   r       -   Firing threshold variance (sigmoid slope)
%   v0      -   Mean firing threshold
%   sigma   -   (optional) Covariance. Should be a vector whose size is the
%               number of states
% 
% Outputs:
%   out     -   Output sigmoid
%
% Note: Plot the non-linearity with the following code:
%{
        params = set_parameters('brunel');       % Chose params.u from a constant value in set_params
        nonlinearity = [];
        count = 0;
        x = -20:0.1:80;
        for i = x
            count = count +1;
            nonlinearity(count) = non_linear_sigmoid(i, params.r, params.v0);
        end
        figure
        plot(x, params.e0*nonlinearity, 'LineWidth', 2);
        box off
        grid on
        ylabel('Output firing rate');
        xlabel('Input membrane potential');
        hold;
        plot([min(x) max(x)],[0.5 0.5]*params.e0,'--k');
        plot([params.v0 params.v0], [0 1]*params.e0,'--k');
%}
%
% Artemio - 2021
function out = non_linear_sigmoid(x, r, v0, varargin)
%% ERF:
    % Check for optional input argument 'sigma'. The default value is an
    % zeros array the same size as x.
    if ~(nargin > 3 && ~isempty(varargin{1})) % nargin <= 3
        input = (x - v0) / (sqrt(2) * r); % Sigma is zero
    else
        input = zeros(size(x));
        sigma = reshape(varargin{1}, [size(x,1) 1]); % Making sure sigma is the same dimmension as x, i.e. the number of states
        for i = 1:size(x,2) % Iterating through first dimmension of x, i.e. the samples.
            input(:,i) = (x(:,i) - v0) ./ (sqrt(2*(r^2 + sigma.^2)));
        end
    end
    out = 0.5*erf(input) + 0.5;
    
% %% Double exponential sigmoid 2^-2^-x
%     if (nargin > 3) && ~isempty(varargin{1})
%         sigma = varargin{1}; % The optional input is the diagonal of the covariance matrix, i.e. variance vector
%         if numel(sigma) > 1
%             sigma = reshape(sigma, [size(x,1) 1]); % If sigma has more than 1 element, shape it the same as x
%         end
%         r_ = r + sigma;
%     else
%         r_ = r; 
%     end
%     out = 2.^-(r_.^-(x-v0)); 

end