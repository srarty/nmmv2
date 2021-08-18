% ANALYTIC_KALMAN_FILTER Implements a Kalman filter where the mean and 
% covariance are propagated from the analytic solution.
% 
% Inputs: y - measurements
%         f - transition function. For a regular Kalman fiter use 
%                 @(x)(F*x), where F is the transition matrix
%         nmm - stores information about the NMM, includes params and 
%                 state-space representation (this is the output of 
%                 nmm_define)
%         H - the observation matrix
%         Q - process covariance
%         R - measurement covariance
%         m0 - mean of prior distribution
%         P0 - covariance of prior distribution
%         integration_method (optional) - 'euler' (Default) or 'runge'
%         nn (optional) - is the integration time step ratio for 
%                 runge-kutta integration. nn = fix(dT/dt), 
%                 dT = sampling, dt = integrtion time step
%         verbose (optional) - If true [default], shows a progress bar 
%                 modal window
%         
% Outputs: x_hat - the posterior mean (the estimated state)
%          P_hat - the posterior covariance
%          K - The Kalman filter gain matrix
%
% Based on Kelvin Layton (Jan 2013)
% Pip Karoly, Sep 2020
% Artemio, Jan 2021
%
function [x_hat, P_hat, K, fe, fi] = analytic_kalman_filter_2(y,f_,nmm,H,Q,R,varargin)
    % Input arguments
    if exist('varargin','var') && length(varargin) > 0
        integration_method = varargin{1};
        nn = 1; if length(varargin) > 1, nn = varargin{2}; end % nn is the integration time step ratio for runge-kutta integration. nn = fix(dT/dt), dT = sampling, dt = integrtion time step
        verbose = true; if length(varargin) > 2, verbose = varargin{3}; end
        % DO_FILTER = true; if length(varargin) > 3, DO_FILTER = varargin{3}; end % Not in use. Use nmm.options.KF_TYPE instead
    else
        integration_method = 'euler'; % 'euler' or 'runge'
    end
    
    % Options (retrieving from nmm struct)
    ALPHA_KF_LBOUND = false; if isfield(nmm.options, 'ALPHA_KF_LBOUND'), ALPHA_KF_LBOUND = nmm.options.ALPHA_KF_LBOUND; end % Lower bound on alpha parameters
    ALPHA_KF_UBOUND = false; if isfield(nmm.options, 'ALPHA_KF_UBOUND'), ALPHA_KF_UBOUND = nmm.options.ALPHA_KF_UBOUND; end % Upper bound on alpha parameters (needs the lower bound to be true)
    KF_TYPE = 'extended'; if isfield(nmm.options, 'KF_TYPE'), KF_TYPE = nmm.options.KF_TYPE; end % String: 'unscented', 'extended' (default) or 'none' 
    if strcmp('none', KF_TYPE), DO_FILTER = false; else, DO_FILTER = true; end % If KF_TYPE is 'none', the estimation calls akf_iterate, but does not compute the KF part
    
    r = nmm.params.r;
    v0 = nmm.params.v0;
    m0 = nmm.x0;
    P0 = nmm.P0;
    
    NSamples = length(y); % Number of samples. Should be equal to Ns, but setting it to lenght of the observed EEG
    NStates = length(m0); % Number of states
    v = mvnrnd(zeros(NStates,1),Q,NSamples)'; % Measurement noise
    
    scale_kf = 1/1; % Factor to which the derivative states (2,4,...) are scaled to mantain all states within a range of magnitudes
    scale_range = [1:2:4];% 5 6 7];%, 5, 6, 7]; % 1:2:NStates
    
    % Initialize mean and covariance.
    % Mean:
    x_hat_init = m0; % to ignore inital transient take the mean of the second half of the test data
    % Covariance (uncomment accordingly):
    % P_hat_init = 10e-2*cov(x(:, size(x,2)/2:end)');
    P_hat_init = P0;%10e-2*eye(N_states);
    % P_hat_init = 10e2*generateSPDmatrix(N_states);
    % P_hat_init( 2*N_syn+1:end, 2*N_syn+1:end ) = eye(N_syn + N_inputs) * 10e-2; % open up the error in parameters

    % Set inital conditions for the KF
    x_hat = zeros(NStates, NSamples);
    P_hat = zeros(NStates, NStates, NSamples);
    P_diag = zeros(NStates, NSamples);

    x_hat(:,1) = x_hat_init;
    P_hat(:,:,1) = P_hat_init;
        
    % Initialize K gain matrix
    K = ones(NStates, NSamples);
    
    % Initialize firing rates vector for tracking (the output of the
    % nonlinearity).
    fe = zeros(1,NSamples);
    fi = zeros(1,NSamples);
    
    % Progress bar
    if strcmp(KF_TYPE, 'unscented'), KF_string = 'Unscented'; else, KF_string = 'Analytic'; end
    if verbose, wbhandle = waitbar(0, [KF_string ' Kalman Filter...']); end
    for n = 1:NSamples-1
        try % Try catch on the whole for loop to prevent issues with the progress bar
            switch KF_TYPE
                case 'unscented'
                    % Compute 1 iteration of the analytic extended kalman filter
                    [x_hat(:,n+1), P_hat(:,:,n+1), K(:,n+1), fe(:,n+1), fi(:,n+1)] = ukf_iterate(x_hat(:,n), P_hat(:,:,n), Q, R, y(n+1), H, ...
                        NStates, f_, scale_kf, scale_range, ALPHA_KF_LBOUND, ALPHA_KF_UBOUND, integration_method, DO_FILTER);
                otherwise
                    % Analytic extended kalman filter. DO_FILTER is true if
                    % 'extended' is chosen, DO_FILTER is false if anything
                    % else is chosen
                    [x_hat(:,n+1), P_hat(:,:,n+1), K(:,n+1), fe(:,n+1), fi(:,n+1)] = akf_iterate(x_hat(:,n), P_hat(:,:,n), Q, R, y(n+1), H, ...
                        NStates, f_, scale_kf, scale_range, ALPHA_KF_LBOUND, ALPHA_KF_UBOUND, integration_method, DO_FILTER);
            end
%             if n == 1, warning('remove next line (debugging)'); end
%             x_hat(6,n+1) =  x_hat(6,n); %20;%
%             x_hat(7,n+1) =  x_hat(7,n); %10;%
        catch ME % Try catch around the whole for loop to make sure we close the progress bar in case there is an error during execution.
            if exist('wbhandle','var')
                delete(wbhandle)
            end
            if strcmp('Couldn''t find the nearest SPD', ME.message)
                disp([ME.message ' at time t = ' num2str(nmm.params.dt * n) ' s | sample = ' num2str(n)]);
            end
            rethrow(ME);
        end
        % Update progress bar
        if verbose, try wbhandle = waitbar(n/NSamples, wbhandle); catch, delete(wbhandle); error('Manually stopped'); end, end
    end
    % Delete progress bar's handle
    if verbose, try delete(wbhandle); catch, error('Oops!');end, end
end