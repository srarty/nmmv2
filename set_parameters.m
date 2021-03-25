% Set parameters for the JR model
%
function params = set_parameters(mode, varargin)

mu = 11; % Default input value

if nargin < 1
    mode = 'alpha';
elseif nargin > 1
    % If the function is called with 2 input arguments, the input (u) is 
    % defined by the second argument
    mu = varargin{1};
end

switch mode
    case 'alpha'
        params.e0 = 2.5;  % max firing rate
%         params.r = 0.56;  % logistic sigmoid slope
        params.r = 3.0285;  % erf sigmoid slope
        params.v0 = 6; % Firing Threshold

        % inverse time constants
        params.decay_e = 100;% (1/ tau_e)
        params.decay_i = 50; % (1/tau_i)

        params.alpha_ei = 3.25;% 3.25;     % Gains (a_ei = excitatory)
        params.alpha_ie = 6.25;%6.25;%6.25;%22;%12.5;  % (a_ie = inhibitory)

        params.u = mu;%11;%220;%15;%11;        % mean input mem potential

        params.dt = 0.001;     % sampling time step   
        
        params.scale = 1; % Scale to fix mismatch in state amplitudes. Not to be confused with the scael in analytic_kalman_filter_2
    otherwise
        error('%s rythm not implemented, sorry!', mode);
end
