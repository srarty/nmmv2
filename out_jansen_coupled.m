close all
clc
%% 

params = set_parameters('alpha');

A = params.alpha_ei;
B = params.alpha_ie;
e0 = params.e0;
decay_e = params.decay_e;
decay_i = params.decay_i;
c=100;
c1=c;
c2=0.8*c;

alpha_i = A*c1*2*e0*decay_e;
alpha_e = B*c2*2*e0*decay_i;

y0=zeros(7,1);
% y0(1) = 6;
% y0(2) = 6;
% y0(3) = 6;
% y0(4) = 6;
y0(5) = 11;
y0(6) = alpha_i; % alpha_i = 162500
y0(7) = alpha_e; % alpha_e = 440000

%% Time axis
tmin=0;
tmax=5;  % s
t=0:1/1200:5;
%% Simulation
% my_jansen = @(t,y) jansen(t,y);
% [tsol,ysol]=ode45(my_jansen,[0 5],y0);
options=odeset('RelTol',1e-3,'AbsTol',1e-4);
[tsol,ysol]=ode45(@(t,y,p) jansen_coupled(t,y,params),t,y0);
% [tsol,ysol]=ode45(@(x,P) nmm_run(nmm, x, P,  'analytic'),t,nmm.x0); % This line doesn't work

%[tsol,ysol]=rkf45(@(t,y) jansen(t,y),0,1,y0,5,1e-3);
% 
output = ysol(:,1);
plot(tsol, output);
