% COMPUTE_PCRB_P Compute the Posterior CRB using Bergman iteration
% 
% Inputs: t - vector of time points
%         f - transition function. For a regular Kalman fiter use 
%                 @(x)(F*x), where F is the transition matrix
%         F - transition matrix function (a function that takes the
%                 current state and returns the Jacobian). For a regular
%                 Kalman fiter use @(x)F, where F is the transition matrix
%         H - observation matrix function. For linear measurement 
%                 use @(x)H, where H is observation matrix
%         Q - process covariance
%         R - measurement covariance
%         m0 - mean of prior distribution
%         P0 - covariance of prior distribution
%         M - number of Monte Carlo samples
%         y - measurements
%         ALPHA_KF_LBOUND - lower bound on alpha parameters at zero (true or false)
%         ALPHA_KF_UBOUND - upper bound on alpha (numerical)
%         KF_TYPE - 'unscented', 'extended' or 'none'
%
% Outputs: pcrb - the posterior CRB
%
% Kelvin Layton - Feb 2013
% Modified by Artemio - 2021
%
function pcrb = compute_pcrb_P_analytic(t,f,F,H,Q,R,m0,P0,M,y,ALPHA_KF_LBOUND, ALPHA_KF_UBOUND, KF_TYPE)

N = length(t);
NStates=length(m0);

% Initialise variables
Fisher=zeros(NStates,NStates,N); % Fisher information matrix
Fisher(:,:,1)=P0;
pcrb=zeros(NStates,N);

% Initalise all trajectories
%
xk=mvnrnd(m0,P0,M)';
Rinv=inv(R);



% Compute the PCRB using a Monte Carlo approximation
nps = []; % Stores the iteration numbers of failed iterations due to P being not positive semidefinite

% Progress bar
wbhandle = waitbar(0, 'Posterior Cramer-Rao Lower Bound...');
for k=2:N
    Fhat = zeros(NStates);
    Rinvhat = zeros(NStates);
    
    v = mvnrnd(zeros(NStates,1),Q,M)';
    
    % Reinitialize P
    P=zeros(NStates,NStates,N);
    P(:,:,1)=P0;
    
    nps = []; % Vector that tracks all the points where there was an error due to the nearest positive semidefinite matrix algorithm
    nps_flag = false; % True when there is an error due to nps
    tic
    parfor i=1:M % parfor
        try
            % Sample the next time point for the current trajectory realisation
%             [xk(:,i), P(:,:,i)] = f(xk(:,i), P(:,:,i));
%             xk(:,i) = xk(:,i) + v(:,i);
%             P(:,:,i) = P(:,:,i) + Q;
            % Kalman filter
            if strcmp('unscented', KF_TYPE)
                % ukf_iterate: unscented kalman filter
                [xk(:,i), P(:,:,i)] = ukf_iterate(xk(:,i), P(:,:,i), Q, R, y(k), H, NStates, f, 1, [], ALPHA_KF_LBOUND, ALPHA_KF_UBOUND, 'euler', true);
            else
                if strcmp('extended', KF_TYPE)
                    DO_FILTER = true;
                else
                    DO_FILTER = false;
                end
                % akf_iterate: analytic (extended) kalman filter
                [xk(:,i), P(:,:,i)] = akf_iterate(xk(:,i), P(:,:,i), Q, R, y(k), H, NStates, f, 1, [], ALPHA_KF_LBOUND, ALPHA_KF_UBOUND, 'euler', DO_FILTER);
            end
            
            % Compute the PCRB terms for the current trajectory realisation
            Fhat = Fhat + F(xk(:,i), P(:,:,i));

            Hmat = H;%H(xk(:,i));
            Rinvhat = Rinvhat + Hmat'*Rinv*Hmat;
        catch E
            if strcmp('MATLAB:erf:notFullReal', E.identifier) ||...
                    strcmp('stats:mvncdf:BadMatrixSigma', E.identifier)
                % P matrix is not positive definite -> Remove iteration
                nps = [nps k];
                nps_flag = true;
%                 break;
            elseif strcmp('Couldn''t find the nearest SPD', E.message)
                disp(['Couldn''t find the nearest SPD at time sample = ' num2str(k)]);
                disp(['Error found while Running PCRB , iteration: ' num2str(i)]);
                % P matrix is not positive definite -> Remove iteration
                nps = [nps k];
                nps_flag = true;
%                 break;
            else
                % If some other error, propagate it
                rethrow(E);
            end
        end
    end
        
    Fhat=Fhat./(M - length(nps));
    Rinvhat=Rinvhat./(M-length(nps));
        
%     Fnorm(k)=norm(Fhat);
%     Feig(k) = min(eig(Fhat));
    
    % Recursively compute the Fisher information matrix
    %
    Fisher(:,:,k) = inv( inv(Fhat*Fisher(:,:,k-1)*Fhat' + Q) + Rinvhat);
    
    % Compute the PCRB at the current time
    %
    pcrb(:,k) = diag(Fisher(:,:,k));
    
    % Update progress bar
    try wbhandle = waitbar(k/N, wbhandle); catch, delete(wbhandle); error('Manually stopped'); end
end

% Delete progress bar's handle
try delete(wbhandle); catch, error('Oops!');end


