% ANALYTIC_KALMAN_FILTER Implements a Kalman filter where the mean and 
% covariance are propagated from the analytic solution.
% 
% Inputs: y - measurements
%         f - transition function. For a regular Kalman fiter use 
%                 @(x)(F*x), where F is the transition matrix
%         F - transition matrix function (a function that takes the
%                 current state and returns the Jacobian). For a regular
%                 Kalman fiter use @(x)F, where F is the transition matrix
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
function [x_hat, P_hat, K, fe, fi] = analytic_kalman_filter_2(y,f_,F_,nmm,H,Q,R,m0,P0,varargin)
    % Input arguments
    if nargin > 9
        integration_method = varargin{1};
        nn = 1; if nargin >10, nn = varargin{2}; end % nn is the integration time step ratio for runge-kutta integration. nn = fix(dT/dt), dT = sampling, dt = integrtion time step
        verbose = true; if nargin >11, verbose = varargin{3}; end
    else
        integration_method = 'euler'; % 'euler' or 'runge'
    end
    
    r = nmm.params.r;
    v0 = nmm.params.v0;
    
    NSamples = length(y); % Number of samples. Should be equal to Ns, but setting it to lenght of the observed EEG
    NStates = length(m0); % Number of states
    v = mvnrnd(zeros(NStates,1),Q,NSamples)'; % Measurement noise
    
    scale = 1/50;%50; % Factor to which the derivative states (2,4,...) are scaled to mantain all states within a range of magnitudes
    scale_range = [1:2:4 5];%, 5, 6, 7]; % 1:2:NStates
    
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
    if verbose, wbhandle = waitbar(0, 'Analytic Kalman Filter...'); end
    for n = 1:NSamples-1
        % Prediction step
        %-------------- start prediction
        if strcmpi('euler', integration_method)
            % Euler Integration
            
%             [x_hat(:,n+1), P_hat(:,:,n+1)] = prop_mean_and_cov(2,NStates,1,...
%                 nmm.A, nmm.B, nmm.C, P_hat_init, x_hat_init, r, v0, Q); 
            [x_hat(:,n+1), P_hat(:,:,n+1), fe(n+1), fi(n+1)] = f_(x_hat(:,n), P_hat(:,:,n)); % [x_hat(:,n+1), P_hat(:,:,n+1)] = f_(x_hat(:,n), zeros(size(P0)));
            P_hat(:,:,n+1) = P_hat(:,:,n+1) + Q;
            
        elseif strcmpi('runge', integration_method)
            % Runge-Kutta Integration
            
            h = 0.25; % step size (number of samples)
            s1 = nan(NStates, NSamples-1);	s2 = nan(NStates, NSamples-1);	s3 = nan(NStates, NSamples-1);	s4 = nan(NStates, NSamples-1);
            p1 = nan(NStates);	p2 = nan(NStates);	p3 = nan(NStates);	p4 = nan(NStates);

            x_ = x_hat(:,n); % Change of variable for ease in notation
            P_ = P_hat(:,:,n); % "

            for i = 1:nn
                [s1, p1] = f_(x_, P_);           % F(t_n, y_n)
                s1 = (s1 - x_);            % Fix the addition (x_ is added within f_(.), but we don't want it to be added)
                p1 = (p1 - P_);

                [s2, p2] = f_(x_ + h*s1/2, P_ + h*p1/2); % F(t_n + h/2, y_n+h*s1/2)
                s2 = (s2 - (x_ + h*s1/2));   
                p2 = (p2 - (P_ + h*p1/2));

                [s3, p3] = f_(x_ + h*s2/2, P_ + h*p2/2); % F(t_n + h/2, y_n+h*s2/2)
                s3 = (s3 - (x_ + h*s2/2)); 
                p3 = (p3 - (P_ + h*p2/2));

                [s4, p4] = f_(x_ + h*s3, P_ + h*p3);     % F(t_n + h, y_n+h*s3)
                s4 = (s4 - (x_ + h*s3));
                p4 = (p4 - (P_ + h*p3));

                x_ = x_ + h*(s1 + 2*s2 + 2*s3 + s4)/6;
                P_ = P_ + h*(p1 + 2*p2 + 2*p3 + p4)/6;
            end

            x_hat(:,n+1) = x_;
            P_hat(:,:,n+1) = P_ + Q;
        else
            error('Invalid integration method');
        end
        %-------------- end prediction
        
        % Scale derivatives (divide by a factor)
        x_hat(scale_range,n+1) = x_hat(scale_range,n+1)./scale;
        P_hat(scale_range, scale_range, n+1) = P_hat(scale_range, scale_range, n+1)./scale;        
        y(n+1) = y(n+1)./scale;
        
        % Update step
        K(:,n+1) = P_hat(:,:,n+1)*H' / ((H*P_hat(:,:,n+1)*H' + R)); % K = P_hat(:,:,n+1)*H' * inv((H*P_hat(:,:,n+1)*H' + R));
        x_hat(:,n+1) = x_hat(:,n+1) + K(:,n+1)*(y(:,n+1)-H*x_hat(:,n+1));
        P_hat(:,:,n+1) = (eye(length(x_hat_init))-K(:,n+1)*H)*P_hat(:,:,n+1); % P_k+ depends on K
        % Use following option to avoid calculating K.
        %P_hat(:,:,n+1) = inv(inv(P_hat(:,:,n+1)) + H'*inv(R)*H); % P_k+ does not depend on K
        
        % Force symmetry on P_hat
        P_hat(:,:,n+1) = (P_hat(:,:,n+1) + P_hat(:,:,n+1)')/2;
        % Check eigenvalues
        [~,flag] = chol(P_hat(:,:,n+1));
        if flag
            % If any is negative, find the nearest Semipositve Definite
            % matrix
            [P_hat(:,:,n+1), k]= nearestSPD(P_hat(:,:,n+1)); % Nearest SPD, Higham (1988) - Parvin's method
            if k == -1
                % Infinite loop in the nearestSPD script. No SPD matrix
                % found
                disp(['Couldn''t find nearest SPD at t = ' num2str(n)]);
            end
        end
        
        % Rescale derivatives back (multiply by scale factor)
        x_hat(scale_range,n+1) = x_hat(scale_range,n+1).*scale;
        P_hat(scale_range, scale_range, n+1) = P_hat(scale_range, scale_range, n+1).*scale; 
        y(n+1) = y(n+1).*scale;
        
        % Update progress bar
        if verbose, try wbhandle = waitbar(n/NSamples, wbhandle); catch, delete(wbhandle); error('Manually stopped'); end, end
    end
    % Delete progress bar's handle
    if verbose, try delete(wbhandle); catch, error('Oops!');end, end
end
