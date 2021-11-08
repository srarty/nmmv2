% Code to analyze Brunel data: Find Brunel's nonlinearity
%
% Artemio - August 2021

clear

%% NMM sigmoid
params = set_parameters('brunel');       % Chose params.u from a constant value in set_params
x = -50:1:300;
nonlinearity = nan(size(x));
for i = 1:numel(x)
    nonlinearity(i) = params.e0 * non_linear_sigmoid(x(i), params.r, params.v0);
end
figure
plot(x,nonlinearity, 'LineWidth', 2);
box off
grid on
ylabel('Output');
xlabel('Input');
hold;
plot([min(x) max(x)],[params.e0*0.5 params.e0*0.5],'--k');
plot([params.v0 params.v0], [0 params.e0*1],'--k');
xlabel('Membrane potential (mV)');
ylabel('Spike rate (normalized)');

%% Load the data
% data_file = 'C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/lfp_last.mat';
% data_file = 'C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/CUBN/lfp_last.mat';

% input_rates = [1:4 5:5:300];
% input_rates = [1:5 10:10:290];
input_rates = [1:5 10:5:400];
membrane_potentials = zeros(1,length(input_rates));
pyramidal_rates = zeros(1,length(input_rates));
for i_rates = 1:length(input_rates)
%     data_file = ['C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/CUBN/lfp_inputRate_' num2str(input_rates(i_rates)) '.mat'];
%     data_file = ['C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/CUBN/iterneurons_lfp_inputRate_' num2str(input_rates(i_rates)) '.mat'];
%     data_file = ['C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/CUBN/inhibitory_input_lfp_inputRate_' num2str(input_rates(i_rates)) '.mat'];
    data_file = ['C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/CUBN/searching for nonlinearity/lfp_refr_x5_inRate_' num2str(input_rates(i_rates)) '.mat'];
    
    load(data_file);
    
    membrane_potentials(i_rates) = 1000*(mean(V_py_single(1,500:end)) - v_rest);
    pyramidal_rates(i_rates) = mean(R_py(500:end))/params.e0;
end

fig = figure;
yyaxis left
scatter(input_rates, pyramidal_rates * params.e0,5,'filled');
xlabel('Input firing rate');
ylabel('Pyramidal firing rate');

%% Fit
% ft = fittype( 'c/(1+exp(-a*(x+b)))+d', 'independent', 'x', 'dependent','y' ); % sigmoid
ft = fittype( '0.5*erf((x - a) / (sqrt(2) * b)) + 0.5', 'independent', 'x', 'dependent', 'y' ); % Error function | a = v0 | b = r
opts = fitoptions(ft);
opts.StartPoint = [6 0]; %[0.5 -10 1 -0.005];
fitresult_Py = fit(input_rates', pyramidal_rates', ft);%, opts);

figure(fig);
yyaxis right
plot(fitresult_Py);
xlabel('Input firing rate');
ylabel('Pyramidal firing rate');
yyaxis left
return;

%% Spike times
Py_cell_number = 1;
movavg_window = 500;

indx_py = round(Py_sp_t(Py_cell_number,:)/(lfp_dt))+1;
indx_in = []; % round(In_sp_t{In_cell_number,:}/(lfp_dt))+1;

%% Normalize membrane potentials (0 to 1)
V_py = (V_py_single(Py_cell_number,:) - v_rest);

V_py = 1000 * V_py;        

V_py_normal = V_py / max(V_py);
R_py_normal = R_py - min(R_py);                 % Remove offset
R_py_normal = R_py_normal / max(R_py_normal);   % Normalize

taum = 200;
taur = 4;
taud = 20;
taul = 10;

% Overlap a spike shape at each spike of the LIF
hh = @(x) -taum/(taud-taur)*(exp(-(x-taul)/taud) - exp(-(x-taul)/taur)) + 17;
% hh = @(x) 70*exp(1./(x*0.5))-70+11;
for i = 1:length(indx_py)
    V_py(indx_py(i) : indx_py(i)+29) = hh(1:30);
end

In_cell_number = 1;
V_in = (V_in_single(In_cell_number,:) - v_rest);
V_in_normal = 1000 * V_in;                      % express in mV
R_in_normal = R_in - min(R_in);                 % Remove offset
R_in_normal = R_in_normal / max(R_in_normal);   % Normalize

% Removing refractory period
V_py_fixed = V_py;
R_py_fixed = R_py_normal(1:9500);
% for i = 1:length(V_py_normal)
%     if V_py_normal(i) >= 19.5
%         if V_py_normal(i + 1) <= 11.001
%             V_py_fixed(i:min(length(V_py_normal), i+20)) = nan;
%         end
%     end
% end
% nan_idx = isnan(V_py_fixed);
% V_py_fixed(nan_idx) = [];
% R_py_fixed(nan_idx) = [];
% V_Py_effective(nan_idx) = [];

% Moving average
% V_py_movavg = smooth(V_py_normal(1,:), 'sgolay', 1);
% V_py_movavg = smooth(V_py_normal(Py_cell_number,:), 50);
V_py_movavg = smooth(V_py_fixed, movavg_window);
V_py_movavg_ = smooth(V_py_normal(Py_cell_number,:), 5);

% Effective Voltage
V_Py_effective = smooth(1000 * (V_Py_effective(1:9500) - v_rest) + 11, 100)'; % 11 is the Vm during refractory period
% V_Py_effective = smooth(1000 * (V_Py_effective(1:9500) - v_rest), 100)';



%% Spike rate
window_size = 1/(lfp_dt * 1e3); % Window size for the moving average (sample). Brunel's window size is 5 ms, here it is the numerator in ms (might be different from brunel's)
% Get spike times in samples
% Convert cells to vector
try
    Py_sp_t = [Py_sp_t{Py_cell_number,:}];
catch E
    if strcmp(E.identifier, 'MATLAB:cellRefFromNonCell')
    % Sometimes it's not saved as cell
       disp('Py_sp_t was already a cell vector.');
    else
        rethrow(E);
    end
end    

spikes_py = V_py_movavg(indx_py);
spikes_in = V_in_normal(1, indx_in);
% Calculate spike rates
spike_rate_py = zeros(1,length(V_py_movavg));
for i = 1:length(spikes_py)-1
%     spike_rate(i) = sum(indx>i & indx<i+window_size); % for loop around up to length(V_py_movavg)
    spike_rate_py(indx_py(i)) = 1/((Py_sp_t(Py_cell_number, i+1) - Py_sp_t(Py_cell_number,i))./1e3); % Instantaneous spike rate (1/time between two spikes)
end
% Inf values are  happened at the "same" time

spike_rate_in = zeros(1,length(V_in_normal));
for i = 1:length(spikes_in)-1
%     spike_rate(i) = sum(indx>i & indx<i+window_size); % for loop around up to length(V_py_movavg)
    spike_rate_in(indx_in(i)) = 1/((In_sp_t(In_cell_number,i+1)-In_sp_t(In_cell_number,i))/1e3); % Instantaneous spike rate (1/time between two spikes)
end

%% Fit
ft = fittype( 'c/(1+exp(-a*(x+b)))+d', 'independent', 'x', 'dependent','y' ); % sigmoid
% ft = fittype( '0.5*erf((x - a) / (sqrt(2) * b)) + 0.5', 'independent', 'x', 'dependent', 'y' ); % Error function
opts = fitoptions(ft);
opts.StartPoint = [0.5 -10 1 -0.005]; % For sigmoid
% opts.StartPoint = [6 0]; % For error function
% opts.Upper(4) = 0;
% opts.Display = 'Off';
% opts.Lower = [0 -10 1 0];
% opts.Upper = [1 10 1 0];
V_py_to_fit = V_py_movavg(501:10000)';
% [fitresult_Py, gof1] = fit([zeros(1,1000) V_Py_effective]', [zeros(1,1000) R_py_fixed]', ft, opts);
[fitresult_Py, gof1] = fit([zeros(1,1000) V_py_to_fit(1:end-1000)]', [zeros(1,1000) R_py_fixed(1:end-1000)]', ft, opts);
[fitresult2, gof2] = fit(V_in_normal', spike_rate_in', ft, opts);

%% Plot
% Plot single membrane potential and spikes
figure
subplot(3,1,1)
% plot(V_py_normal(Py_cell_number,:))
plot(V_py_fixed)
hold on;
plot(V_py_movavg)
% plot(indx_py, spikes_py, 'ob');
% plot(V_in_normal(1,:))
% plot(indx_in, spikes_in, 'or');
subplot(3,1,2)
% plot(spike_rate_py);
plot(R_py);
hold
% plot(spike_rate_in);
xlim([0 length(V_py_single)]);
subplot(3,1,3)
plot(V_Py_effective)
hold
plot(V_py_movavg)

% Plot sigmoid
figure
plot(fitresult_Py, 'b');
hold on;
% plot(fitresult_Py_, '--b');
% plot(fitresult2,'r');
% scatter(V_py_movavg', spike_rate_py', 'b');
% scatter(V_py_movavg', R_py', 'b');
% scatter(V_py_movavg, R_py_fixed', 'b');
scatter(V_py_to_fit(1:end-1000), R_py_fixed(1:end-1000)', 'b');
% scatter(V_in_normal', spike_rate_in', 'r');
xlabel('Membrane potential (mV)')
ylabel('Spike rate (normalized)')

