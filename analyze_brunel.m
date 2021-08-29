% Code to analyze Brunel data.
%
% Artemio - August 2021

clear

%% NMM sigmoid
params = set_parameters('brunel');       % Chose params.u from a constant value in set_params
x = -5:0.1:20;
nonlinearity = nan(size(x));
for i = 1:numel(x)
    nonlinearity(i) = non_linear_sigmoid(x(i), params.r, params.v0);
end
figure
plot(x,nonlinearity, 'LineWidth', 2);
box off
grid on
ylabel('Output');
xlabel('Input');
hold;
plot([min(x) max(x)],[0.5 0.5],'--k');
plot([params.v0 params.v0], [0 1],'--k');
xlabel('Membrane potential (mV)');
ylabel('Spike rate (normalized)');

%% Load the data
data_file = 'C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/lfp_last.mat';
load(data_file); 
    
%% Normalize membrane potentials (0 to 1)
V_py = (V_py_single(1,:) - v_rest);
V_py_normal = 1000 * V_py;        
R_py_normal = R_py - min(R_py);                 % Remove offset
R_py_normal = R_py_normal / max(R_py_normal);   % Normalize

V_in = (V_in_single(1,:) - v_rest);
V_in_normal = 1000 * V_in;        % express in mV
R_in_normal = R_in - min(R_in);                 % Remove offset
R_in_normal = R_in_normal / max(R_in_normal);   % Normalize

%% Spike rate
window_size = 1/(lfp_dt * 1e3); % Window size for the moving average (sample). Brunel's window size is 5 ms, here it is the numerator in ms (might be different from brunel's)
% Get spike times in samples
indx = round(V_sp_t/(lfp_dt*1e3))+1;
% spikes = V_in_single(1,round(V_sp_t/(lfp_dt*1e3))+1);
spikes = V_in_normal(1,round(V_sp_t/(lfp_dt*1e3))+1);
% Calculate spike rates
spike_rate = zeros(1,length(V_py_normal));
for i = 1:length(spikes)-1
%     spike_rate(i) = sum(indx>i & indx<i+window_size); % for loop around up to length(V_py_normal)
    spike_rate(indx(i)) = 1/((V_sp_t(i+1)-V_sp_t(i))/1e3); % Instantaneous spike rate (1/time between two spikes)
end

%% Fit
ft = fittype( 'c/(1+exp(-a*(x+b)))+d', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions(ft);
opts.StartPoint = [0.5 -10 1 -0.005];
% opts.Display = 'Off';
% opts.Lower = [0 -10 1 0];
% opts.Upper = [1 10 1 0];
[fitresult1, gof1] = fit(V_in_normal', spike_rate', ft, opts);
% [fitresult2, gof2] = fit(V_py_normal', R_py_normal', ft, opts);

%% Plot
% Plot single membrane potential and spikes
figure
subplot(2,1,1)
% plot(smooth(V_in_single(1,:),10))
plot(V_in_normal(1,:))
hold on;
plot(indx, spikes, 'o');
subplot(2,1,2)
plot(spike_rate);
xlim([0 length(V_in_single)]);

% Plot sigmoid
figure
plot(fitresult1);
hold on;
% plot(fitresult2,'b');
% scatter(V_py_normal', spike_rate', 'r');
scatter(V_in_normal', R_py_normal', 'b');
xlabel('Membrane potential (mV)')
ylabel('Spike rate (normalized)')

