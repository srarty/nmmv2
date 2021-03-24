%% Simulate 4 state NMM and EKF estimates. Calculate the CRB
% Can be modified to more states by changing NStates. And the model
% functions, e.g. model_NM = 4 states, model_NM_6thOrder = 6 states

close all
clear

%% Options --
NStates = 4; % Number of states
NInputs = 1; % Number of external inputs (u)
NParams = 2; % Number of synaptic strength parameters (alpha_ie, alpha_ei, etc...)
NAugmented = NStates + NInputs + NParams; % Total size of augmented state vector

ESTIMATE = true;    % Run the forward model and estimate (ESTIMATE = true), or just forward (ESTIMATE = false)
REMOVE_DC = false;  % Remove DC offset from simulated observed EEG
ADD_NOISE = true;	% Add noise to the forward model
FIX_PARAMS = false;	% Fix input and alpha parameters to initial conditions

PCRB = false;       % Compute the PCRB (true or false)
REAL_DATA = false;  % True to load Seizure activity from neurovista recordings, false to generate data with the forward model
TRUNCATE = false;   % If true, the real data from recordings is truncated from sample 1 to 'N'
SCALE_DATA = true;  % Scale Raw data to match dynamic range of the membrane potentials in our model

relativator = @(x)sqrt(mean(x.^2,2));% @(x)(max(x')-min(x'))'; % If this is different to 1, it calculates the relative RMSE dividing by whatever this value is.
% -----------

%% Initialization
% params = set_parameters('alpha', mu); % Set params.u from the input argument 'mu' of set_params
params = set_parameters('alpha');       % Chose params.u from a constant value in set_params

N = 5000;             	% number of samples
dT = params.dt;         % sampling time step (global)
dt = 1*dT;            	% integration time step
nn = fix(dT/dt);      	% (used in for loop for forward modelling) the integration time step can be small that the sampling (fix round towards zero)
t = 0:dt:(N-1)*dt;

% model structure
% ~~~~~~~~~~~~~~~~
%           u
%           |  __
%           | /  \
%           |/   a_ie
%      |    |     |
%      v    E     I   ^
%           |     |   |  direction of info
%           |\   a_ei
%           | \   |
%           |  \_/
%           v
%
% Populations:
%   E - excitatory(pyramidal)
%   I - inhibitory
% Gains:
%   a_ei - connectivity strength from excitatory to inhibitory (alpha_ei)
%   a_ie - connectivity strength from inhibitory to excitatory (alpha_ie)
%

u = params.u;
alpha = [params.alpha_ie; params.alpha_ei]; % Parameters in the augmented state vector. Synaptic strengths

% Initialise trajectory state
x0 = zeros(NAugmented,1); % initial state
x0(1:NStates) = mvnrnd(x0(1:NStates),10^1*eye(NStates));
x0(NStates+1:end) = [u; alpha];

% Initialize covariance matrix
P0 = 1e2*eye(NAugmented);
P = zeros(NAugmented, NAugmented, N);
P(:,:,1) = P0;

% Initialize vector to store firing rates (output of the sigmoid)
f_i = zeros(1,N); % Firing rate of the inhibitory neurons
f_e = zeros(1,N); % Firing rate of the excitatory neurons

% Define the model
nmm = nmm_define(x0, P0, params);
x0 = nmm.x0;
x = zeros(NAugmented,N);
x(:,1) = x0;

% Transition model
f = @(x)nmm_run(nmm, x, [], 'transition');
F = @(x)nmm_run(nmm, x, [], 'jacobian');
% Analytic
f_ = @(x,P)nmm_run(nmm, x, P,  'analytic');
F_ = @(x,P)nmm_run(nmm, x, P,  'jacobian');

%% Generate trajectory
% Euler integration
for n=1:N-1
    x(:,n+1) = f(x(:,n)); % Zero covariance
    if FIX_PARAMS, x(NStates+1:end,n+1) = x0(NStates+1:end); end % Fix the parameters (u and alpha)
end

% Calculate noise covariance based on trajectory variance over time??
%   Why noise on all states?
Q = 10^-3.*diag((0.4*std(x,[],2)*sqrt(dt)).^2); % The alpha drift increases with a large covariance noise (Q)
Q(NStates+1 : end, NStates+1 : end) =  10e-1*eye(NAugmented - NStates);

% Initialise random number generator for repeatability
rng(0);

v = 10e-3.*mvnrnd(zeros(NAugmented,1),Q,N)';

% Generate trajectory again with added noise
% Euler-Maruyama integration
for n=1:N-1
    [x(:,n+1), ~, f_i(n+1), f_e(n+1)] = f(x(:,n)); % Propagate mean
    
    if (FIX_PARAMS), x(NStates+1:end,n+1) = x0(NStates+1:end); end % Fixing the parameters
    x(:,n+1) = x(:,n+1) + (ADD_NOISE * v(:,n)); % Add noise if the ADD_NOISE option is true.
end

% Observation function (H = [1 0 0 0 1 0 0]) <- state x1 and parameter u
% are in the observation matrix.
H = zeros(1, NAugmented);
H(1) = 1;        
H(NStates + 1) = 1;
H = H.*1; % Scale the observation matrix if needed

R = 1^-3*eye(1);

w = mvnrnd(zeros(size(H,1),1),R,N)';
y = H*x + w;

if REMOVE_DC
    % Remove DC offset from simulation (Observed EEG)
    y = y - mean(y(length(y)/2:end));
end

if ~ESTIMATE
    %% Plot only the forward model results (generated trajectory)
    % Plot x(1) and ECoG
    figure
    ax1=subplot(211);
    plot(t,x(1,:)');
    ylabel('State 1');
    ax2=subplot(212);
    plot(t,y);
    ylabel('ECoG (mV)');
    xlabel('Time (s)');
    linkaxes([ax1 ax2],'x');

    % Plot all 4 states
    figure
    axs = nan(NStates,1); % Axes handles to link subplots x-axis
    for i = 1:NStates
        axs(i) = subplot(NStates, 1, i);
        plot(t(1:min(length(t),size(x,2))),x(i,1:min(length(t),size(x,2)))');
        ylabel(['State ' num2str(i)]);
    end
    linkaxes(axs, 'x');
    xlabel('time');
    
    % Plot external input and augmented parameters
    figure
    axs = nan(NAugmented - NStates,1); % Axes handles to link subplots x-axis
    for i = 1:NAugmented - NStates
        axs(i) = subplot(NAugmented - NStates, 1, i);
        plot(t(1:min(length(t),size(x,2))),x(NStates + i,1:min(length(t),size(x,2)))');
        ylabel(['Parameter ' num2str(i)]);
    end
    linkaxes(axs, 'x');
    xlabel('Time (s)');
    
    % Firing rates (Output of the nonlinearity)
    figure
    ax1 = subplot(2,1,1);
    plot(t, f_e);
    title('Sigmoid function output');
    ylabel('f_e');
    ax2 = subplot(2,1,2);
    plot(t, f_i);
    ylabel('f_i');
    linkaxes([ax1 ax2],'x');
    xlabel('Time (s)');
   
    return
end
    
if REAL_DATA
    % Load real data from .mat :
    load('./data/Seizure_1.mat');  % change this path to load alternative data
    y = Seizure(:,1)';
    params.dt = T/length(y);
    if TRUNCATE
        y = y(1:N);
    end
    if SCALE_DATA
        y = y*6/50;
    end
    t = params.dt:params.dt:params.dt*length(y);
end

%% Run EKF for this model
% Prior distribution (defined by m0 & P0)
m0 = mean(x(:,ceil(size(x,2)/2):end),2);%mean();% x0;
m0(5) = mean(y(ceil(size(y,2)/2):end));
% m0(6) = 10*rand();
% m0(7) = 10*rand();
% P0 = 1e2*eye(NAugmented); % P0 will use the same initial value as the
% forward model
% P0(NAugmented - NParams + 1 : end, NAugmented - NParams + 1 : end) = 1e3*eye(NParams);

% Apply EKF filter
[m, Phat, ~, fi_exp, fe_exp] = analytic_kalman_filter_2(y,f_,[],nmm,H,Q,R,m0,P0,'runge');

% y_ekf = H*m_;% + w;
y_analytic = H*m;% + w;

%% Plot results
%
if ~REAL_DATA
    % Plot x(1) and ECoG
    figure
    ax1=subplot(211);
    plot(t,x(1,:)'); hold on;
    plot(t,m(1,:)','--'); % EKF
%     plot(t,m_(1,:)','--'); % Analytic, rung kuta
    % plot(t,m__([1],:)','--'); % Analytic euler
    legend({'Actual','Estimation'});
    ylabel('State 1');
    ax2=subplot(212);
    plot(t,y); hold on;
%     plot(t,y_ekf, '--');
    plot(t,y_analytic, '--');
    legend({'Observed EEG', 'Estimated EEG (Analytic)'});
    % plot(t,pcrb(1,:)')
    % legend({'CRLB'})
    ylabel('ECoG (mV)');
    xlabel('Time (s)');
    linkaxes([ax1 ax2],'x');

    % Plot all 4 states
    figure
    axs = nan(NStates,1); % Axes handles to link subplots x-axis
    for i = 1:NStates
        axs(i) = subplot(NStates, 1, i);
        plot(t(1:min(length(t),size(x,2))),x(i,1:min(length(t),size(x,2)))'); hold on;
        plot(t,m(i,:)','--');
%         plot(t,m_(i,:)','--');
    %     plot(t,m__(i,:)','--');
        ylabel(['State ' num2str(i)]);
    end
    linkaxes(axs, 'x');
    legend({'Simulation', 'Estimation'});
    xlabel('time');
    
    % Plot external input and augmented parameters
    figure
    axs = nan(NAugmented - NStates,1); % Axes handles to link subplots x-axis
    for i = 1:NAugmented - NStates
        axs(i) = subplot(NAugmented - NStates, 1, i);
        plot(t(1:min(length(t),size(x,2))),x(NStates + i,1:min(length(t),size(x,2)))'); hold on;
        plot(t,m(NStates + i,:)','--');
%         plot(t,m_(i,:)','--');
    %     plot(t,m__(i,:)','--');
        ylabel(['Parameter ' num2str(i)]);
    end
    linkaxes(axs, 'x');
    xlabel('time');
    legend({'Simulation', 'Estimation'});
    
    % Firing rates (Output of the nonlinearity)
    figure
    ax1 = subplot(2,1,1);
    plot(t, f_e);
    hold
    plot(t, fe_exp, '--');
    title('Sigmoid function output');
    ylabel('f_e');
    ax2 = subplot(2,1,2);
    plot(t, f_i);
    hold
    plot(t,fi_exp, '--');
    ylabel('f_i');
    linkaxes([ax1 ax2],'x');
    xlabel('Time (s)');
        
    % Covariance (Estimation)
    figure
    % plt = @(x,varargin)plot(t,squeeze(x)./max(abs(squeeze(x))),varargin{1});
    plt = @(x,varargin)plot(t,squeeze(x),varargin{1});
    for i = 1:NAugmented
        subplot(2,4,i)
        % Forward model
        plt(P(i,i,:),'-'); hold on;
        % Estimation
        plt(Phat(i,i,:),'--');hold on;
%         plt(Phat_(i,i,:),'--');
    end
    legend({'Analytic', 'EKF'});
    subplot(2,4,1);
    title('Covariance matrix (P) - Diagonal');
else
    % If estimating real data
    figure
    ax1=subplot(211);
    plot(t,m([1],:)','-'); hold on; % Analytic KF
    legend({'Prediction'});
    ylabel('State 1 (Vm)');
    
    ax2=subplot(212);
    plot(t,y_analytic, '--', 'LineWidth', 2);hold on
    plot(t,y);
    legend('Prediction','Observed ECoG'); % legend('EKF', 'Analytic (euler)','Observed EEG');
    linkaxes([ax1 ax2],'x');
    ylabel('ECoG');
    xlabel('Time (s)');
    
    
    % Plot the 4 states
    figure
    axs = nan(NStates,1); % Axes handles to link subplots x-axis
    for i = 1:NStates
        axs(i) = subplot(NStates, 1, i);
        plot(t,m(i,:)','-');hold on;
        ylabel(['State ' num2str(i)]);
    end
    linkaxes(axs, 'x');
    xlabel('Time (s)');

    % Plot the 3 parameters
    figure
    axs = nan(NAugmented-NStates,1); % Axes handles to link subplots x-axis
    for i = NStates+1:NAugmented
        axs(i-NStates) = subplot(NAugmented-NStates, 1, i-NStates);
        plot(t,m(i,:)','-');hold on;
        ylabel(['Parameter ' num2str(i-NStates)]);
    end
    linkaxes(axs, 'x');
    xlabel('Time (s)');
    
    % Firing rates (Output of the nonlinearity)
    figure
    ax1 = subplot(2,1,1);
    plot(t, fe_exp, '-');
    title('Sigmoid function output (prediction)');
    ylabel('f_e');
    ax2 = subplot(2,1,2);
    plot(t,fi_exp, '-');
    ylabel('f_i');
    linkaxes([ax1 ax2],'x');
    xlabel('Time (s)');
    
    % Covariance (Estimation)
    figure
    % plt = @(x,varargin)plot(t,squeeze(x)./max(abs(squeeze(x))),varargin{1});
    plt = @(x,varargin)plot(t,squeeze(x),varargin{1});
    plt_ = @(x,varargin)plot(t,squeeze(x),varargin{1});
    for i = 1:NAugmented
        subplot(2,4,i)
        % Estimation
        plt(Phat(i,i,:),''); hold on;
        % Forward model
%         plt_(Phat_(i,i,:),'--');
    end
%     legend({'EKF' 'Analytic'});
    legend({'Analytic'});
    subplot(2,4,1);title('Covariance matrix (P) - Diagonal');

end

%% Compute the posterior Cramer-Rao bound (PCRB)
if ~PCRB
    return
else
    M = 100;    % Number of Monte Carlo samples
%     pcrb = sqrt(compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0,M)); % Square root to compare it to the Root Mean Square Error
    pcrb_analytic = sqrt(compute_pcrb_P_analytic(t,f_,F_,@(x)H,Q,R,m0,P0,M)) ./ relativator(x); % Divided by the range of the data to calculate the relative rmse % Square root to compare it to the Root Mean Square Error
    % pcrb = compute_pcrb_P(t,f_,F,@(x)H,Q,R,m0,P0,M); % f_ for analytic KF
    % pcrbx5 = compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0.*5,M); % Changed initial condition, multiply P0 by 5
    % pcrbd5 = compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0./5,M); % Changed initial condition, divide P0 by 5
end

%% Compute the MSE of the extended Kalman filter
num_trials = 100;
if ~REAL_DATA
    error = zeros(NAugmented,N);
    error_ = zeros(NAugmented,N);
    % error__ = zeros(NAugmented,N);
else
    error = zeros(size(y));
    error_ = zeros(size(y));
end
nps = 0; % Non-positive semidefinite P matrix, iteration counter for removal
% parfor r=1:num_trials
% To avoid calculating the new trajectory every iteration. Comparing to the "real" x value generated above
z = y; 
% Progress bar
wbhandle = waitbar(0, 'Calculating MSE...');
for r=1:num_trials
    
%     % Create new trajectory realisation
%     %
%     v = mvnrnd(zeros(1,NAugmented)',Q,N)';
%     x = zeros(NAugmented,N);
%     x(:,1)=mvnrnd(m0,P0)';
%     for i=NAugmented:N
%         x(:,i) = f(x(:,i-1)) + v(:,i);
%     end
% 
%     % Generate new observations 
%     %
%     w = mvnrnd(zeros(1,size(H,1)),R,N)';
%     z = H*x + w;
    
    % Apply EKF filter
%     m = extended_kalman_filter_2(z,f,F,H,Q,R,m0,P0);
    try
        m_ = analytic_kalman_filter_2(z,f_,F_,nmm,H,Q,R,m0,P0,'euler',1,false);
%         m__ = analytic_kalman_filter_2(z,f_,F_,H,Q,R,m0,P0,'runge');
    catch E
        if strcmp('MATLAB:erf:notFullReal', E.identifier) ||...
                strcmp('stats:mvncdf:BadMatrixSigma', E.identifier)
            % P matrix is not positive definite -> Remove iteration
            nps = nps + 1; % Fails counter
            % Error is 0 for failed iterations, nps is subtracted from the
            % total number of iterations to calculate MSE.
            % Continue with next iteration of the for loop without adding
            % any error. This iteration won't affect the MSE.
            continue; 
        else
            % If some other error, propagate it
            rethrow(E);
        end
    end

    if ~REAL_DATA
        % Accumulate the estimation error
%         error = error + (x-m).^2;
        error_ = error_ + (x-m_).^2;
    %     error__ = error_ + (x-m__).^2;
    else
%         y_analytic = H*m;% + w;
        y_ekf = H*m_;% + w;
        
        error_ = error_ + (y-y_ekf).^2;
%         error = error + (y-y_analytic).^2;
    end
    % Update progress
    try wbhandle = waitbar(r/num_trials, wbhandle); catch, delete(wbhandle); error('Manually stopped'); end
end
try delete(wbhandle); catch, error('Oops!');end
% Calculate the mean squared error
%
num_trials = num_trials - nps; % Subtract failed iterations
if ~REAL_DATA
%     rmse = sqrt(error ./ num_trials) ./ relativator(x); % Divided by the range of the data to calculate the relative rmse
    rmse_ = sqrt(error_ ./ num_trials) ./ relativator(x);
    % mse__ = error__ ./ num_trials./relativator(x);
    % rmse = sqrt(error ./ num_trials);
else
    rmse_ = sqrt(error_ ./ num_trials) ./ relativator(y); % Divided by the range of the data to calculate the relative rmse
end
%% Plot MSE and the PCRB vs Time
%
if PCRB
    figure('Name', 'NMM - EKF vs CRB')
    if ~REAL_DATA
        for i = 1:NAugmented
            subplot(2,4,i)
            semilogy(rmse_(i,:),'.-'); hold on;
        %     semilogy(rmse(i,:),'x-');
%             semilogy(rmse_(i,:),'.-');
        %     semilogy(mse__(i,:),'.-');
%             semilogy(pcrb(i,:),'.-');
            semilogy(pcrb_analytic(i,:),'.-');
            grid on;
            xlabel('Time (s)');
            ylabel(['RMSE state ' num2str(i)]);
            hold off;
        end
    else
        for i = 1:NAugmented
            subplot(2,4,i);
            semilogy(pcrb_analytic(i,:),'.-'); hold on;
%             semilogy(pcrb(i,:),'.-');
            grid on;
            xlabel('Time (s)');
            ylabel(['RMSE state ' num2str(i)]);
            legend({'PCRB', 'PCRB Analytic'});
            ylim([10^-1 10^5]);
        end
        
        figure
        semilogy(rmse_,'.-'); hold on;
%         semilogy(rmse_,'.-');
        ylim([10^-6 10^6]);
        grid on;
    end
    % legend({'RMSE (EKF)', 'RMSE (Analytic - Euler)', 'RMSE (RK)', 'PCRB', 'PCRB Analytic'});
    legend({'RMSE', 'PCRB'});
end

%% Plot average RMSE vs CRB 
figure('Name', 'Mean Vm - EKF vs CRB')
color = [0.1,0.6,0.7];
%semilogy(rmse(i,:),'x-');
hAx = plot(mean(rmse_,2),'o',...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',color);
hold on;

if PCRB
    bar(mean(pcrb_analytic,2), 'FaceColor', color+0.1);
%     bar(mean(pcrb,2), 'FaceColor', color-0.1,'BarWidth', 0.4);
end

% semilogy(mean(rmse_,2),'o',...
%     'MarkerSize',10,...
%     'MarkerEdgeColor','k',...
%     'MarkerFaceColor',color+0.2);
% semilogy(mean(mse__,2),'^',...
%     'MarkerSize',10,...
%     'MarkerEdgeColor','k',...
%     'MarkerFaceColor',color+0.2);
yt = get(gca, 'YTick'); YTickLabels = cellstr(num2str(round(log10(yt(:))), '10^%d')); % Format y axis in log scale
grid on;
if PCRB
    legend({'RMSE (Analytic)', 'PCRB Analytic','PCRB','RMSE (Euler)', 'RMSE (RK)'});
else
    legend({'RMSE (EKF)', 'RMSE (Euler)'});
end    
uistack(hAx,'top');
xlabel('Time (s)');
ylabel('RMSE (%)');
% ylim([10^-3 10^0])

%% Plot the evolution of the CRB
if PCRB
    figure('Name', 'CRB Convergence')
    for i = 1:NAugmented
        StateNum = i;

%         pcrb_mvnavg = movmean(pcrb,1,2); % Sqrt of PCRB original initial conditions
        pcrb_analytic_mvnavg = movmean(pcrb_analytic,1,2); % Sqrt of PCRB original initial conditions
        % pcrbx5_sqrt = movmean(sqrt(pcrbx5),1,2); % Sqrt of PCRB when P0 is 5 times bigger
        % pcrbd5_sqrt = movmean(sqrt(pcrbd5),1,2); % Sqrt of PCRB when P0 is 5 times smaller
        range_ = 10:length(t)/2; % Range of values to plot to avoid the initial overshoot and to remove the entries beyond convergence

        subplot(2,4,i)
%         plot(t(range_), pcrb_mvnavg(StateNum, range_), 'LineWidth', 2);
%         hold on;
        plot(t(range_), pcrb_analytic_mvnavg(StateNum, range_), 'LineWidth', 2);
        % plot(t(range_), pcrbx5_sqrt(StateNum, range_), '--', 'LineWidth', 2);
        % plot(t(range_), pcrbd5_sqrt(StateNum, range_), '--', 'LineWidth', 2);
        % legend({'P0 = 10000I', 'P0 = 50000I', 'P0 = 2000I'});
        grid on;
        title(['State: ', num2str(StateNum)]);
        xlabel('Time (s)');
        ylabel('BCRB(mV)');
        % xlim([0 5]);
    end
    legend('PCRB', 'PCRB (Analytic)');
end
%%
if PCRB
    figure
%     subplot(2,1,1)
    if ~REAL_DATA
        semilogy(t,sum(rmse_),'x-')
    else
        semilogy(t,rmse,'x-')
    end
    hold on;
    % semilogy(t,sum(mse__),'x-')
    semilogy(t,sum(pcrb_analytic),'o-');
    grid on;
    xlabel('Time (s)');
    ylabel('RMSE');
    legend('RMSE', 'PCRB');
    
%     subplot(2,1,2)
%     if ~REAL_DATA
%         semilogy(t,sum(rmse_),'x-'); hold on;
%     else
%         semilogy(t,rmse_,'x-'); hold on;
%     end
%     semilogy(t,sum(pcrb_analytic),'o-');
%     legend('RMSE Analytic', 'PCRB Analytic');
%     grid on;
%     xlabel('Time (s)');
%     ylabel('RMSE');
%     legend('RMSE', 'PCRB');
end