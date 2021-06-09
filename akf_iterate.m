% AKF_ITERATE performs one iteratin of the analytic kalman filter. 
% It is called from analytic_kalman_filter_2.m, but it can be called from
% any script after each iteration of the inverted NMM.
%
% Inputs: 
%       x_hat   - a posteriori estimated state vector in time t + 1
%       P_hat   - a posteriori estimated P matrix time t + 1
%       K       - gain of the KF
%       fe      - input firing rate to excitatory neurons
%       fi      - output   " ...
%
% Outputs: 
%       x           - states at time t
%       P           - covariance matrix at time t
%       Q           - State noise covariance
%       R           - observation noise covariance?
%       y           - recording at time t
%       H           - observation function (vector)
%       NStates     - number of states (size of x)
%       f_          - (function handle) Neural Mass Model transition 
%                       function
%       scale_kf    - multiplier for the states defined by [scale_range]
%                       to scale only some states before and rescale after 
%                       the Kalman Filter
%       scale_range - vector with states to scale (see [scale_kf])
%       ALPHA_KF_LBOUND - (boolean) if true, lower bound on alpha to 0
%       ALPHA_KF_UBOUND - (integer) upper bound on alpha
%       i_method    - integration method (string) 'euler' or 'runge'
%       DO_FILTER   - if 'none', the KF doesn't run. It returns the
%                       prediction without KF
%
%
% Artemio - April 2021
function [x_hat, P_hat, K, fe, fi] = akf_iterate(x, P, Q, R, y, H, NStates,...
      f_, scale_kf, scale_range,ALPHA_KF_LBOUND, ALPHA_KF_UBOUND, i_method, DO_FILTER)

% Prediction step
%-------------- start prediction
if strcmpi('euler', i_method)
    % Euler Integration
    [x_hat, P_hat, fe, fi] = f_(x, P); % [x_hat, P_hat] = f_(x, zeros(size(P0)));
    P_hat = P_hat + Q;
    
elseif strcmpi('runge', i_method)
    % Runge-Kutta Integration
    h = 0.25; % step size (number of samples)
    s1 = nan(NStates);	s2 = nan(NStates);	s3 = nan(NStates);	s4 = nan(NStates);
    p1 = nan(NStates);	p2 = nan(NStates);	p3 = nan(NStates);	p4 = nan(NStates);
    
    x_ = x; % Change of variable for ease in notation
    P_ = P; % "
    
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
    
    x_hat = x_;
    P_hat = P_ + Q;
else
    error('Invalid integration method');
end
%-------------- end prediction
K = zeros(7,1);
if DO_FILTER
    % Scale derivatives (divide by a factor)
    x_hat(scale_range) = x_hat(scale_range)./scale_kf;
    P_hat(scale_range, scale_range) = P_hat(scale_range, scale_range)./scale_kf;
    y = y./scale_kf;

    % Update step
    K = P_hat*H' / ((H*P_hat*H' + R)); % K = P_hat*H' * inv((H*P_hat*H' + R));
    x_hat = x_hat + K*(y(:)-H*x_hat);
    P_hat = (eye(length(x_hat))-K*H)*P_hat; % P_k+ depends on K
    % Use following option to avoid calculating K.
    %P_hat = inv(inv(P_hat) + H'*inv(R)*H); % P_k+ does not depend on K

    % Force symmetry on P_hat
    P_hat = (P_hat + P_hat')/2;
    % Check eigenvalues
    [~,flag] = chol(P_hat);
    if flag
        % If any is negative, find the nearest Semipositve Definite
        % matrix
        try
            [P_hat, k]= nearestSPD(P_hat); % Nearest SPD, Higham (1988) - Parvin's method
        catch ME
            if strcmp('MATLAB:svd:matrixWithNaNInf', ME.identifier)
                disp(['Error during iteration: ?']);% num2str(n)]);
            end
            rethrow(ME);
        end
        if k == -1
            % Infinite loop in the nearestSPD script. No SPD matrix
            % found
            %disp(['Couldn''t find nearest SPD']);% at t = ?']);% num2str(n)]);
            error('Couldn''t find the nearest SPD');
        end
    end


    % Force alpha-parameters bounds above zero
    if ALPHA_KF_LBOUND
        %             if x_hat(6,n+1) < 0
        %                 x_hat(6,n+1) = x_hat(6,n+1) - 2 * x_hat(6,n+1);
        %                 x_hat(7,n+1) = x_hat(7,n+1) - 2 * x_hat(6,n+1); % Adding the same ammount to the other parameter
        %             end
        %             if x_hat(7,n+1) < 0
        %                 x_hat(6,n+1) = x_hat(6,n+1) - 2 * x_hat(7,n+1);
        %                 x_hat(7,n+1) = x_hat(7,n+1) - 2 * x_hat(7,n+1); % Adding the same ammount to the other parameter
        %             end
        % https://stackoverflow.com/questions/38736481/bounded-kalman-filter
        % Simon (2010) Kalman Filtering with state constraints: a survey of
        % linear and nonlinear algorithms
        amax = ALPHA_KF_UBOUND;
        amin = 0;
        if ALPHA_KF_UBOUND
            % Upper and lower
            x_hat(6) = min(amax, max(amin, x_hat(6)));
            x_hat(7) = min(amax, max(amin, x_hat(7)));

            % Hardcoded bounds on membrane potentials
    %         x_hat(1) = min(30, max(-10, x_hat(1)));
    %         x_hat(3) = min(30, max(-10, x_hat(3)));
        else
            % Only lower bound on zero
            x_hat(6) = max(amin, x_hat(6));
            x_hat(7) = max(amin, x_hat(7));
        end
    end

    % Rescale derivatives back (multiply by scale factor)
    x_hat(scale_range) = x_hat(scale_range).*scale_kf;
    P_hat(scale_range, scale_range) = P_hat(scale_range, scale_range).*scale_kf;
    y = y.*scale_kf;
end % If statement (DO_FILTER)
end % End function