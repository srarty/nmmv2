%% Jansen model Coupled
function dy = jansen_coupled(t,y, params)

A = params.alpha_ei;
B = params.alpha_ie;
decay_e = params.decay_e;
decay_i = params.decay_i;
taue = 1/decay_e;
taui = 1/decay_i;
% taud=0.0303;

% c3=0.25*c;
% c4=0.25*c;
KK=25; % ? don't know what this does
v0 = params.v0; % mV
e0= params.e0; % 1/s
r = params.r; % 1/mV
II=71; % ? Sames as KK, i.e. I don;t know
dy=zeros(7,1);

% S5=S((c3*y(1)),v0,e0,r);
% S6=S((y(1)),v0,e0,r);
% 
% S7=S((y(6)),v0,e0,r);
% S8=S((c1*y(5)),v0,e0,r);
% S9=S((c3*y(5)),v0,e0,r);
% S10=S((y(5)),v0,e0,r);
% %Pyramidal
% dy(1)=y(9);
% dy(9)=(A/taue)*S3-(2/taue)*y(9)-(1/taue^2)*y(1);
% %Excitatory
% dy(2)=y(10)-y(11)+y(16);
% dy(10)=(A/taue)*(II+c2*S4)-(2/taue)*y(10)-(1/taue^2)*(y(2)+y(3)-y(8));
% %Inhibitory
% dy(3)=y(11);
% dy(11)=(B/taui)*c4*S5-(2/taui)*y(11)-(1/taui^2)*y(3);
% %Second column
% dy(4)=y(12);
% dy(12)=(A/taud)*KK*S6-(2/taud)*y(12)-(1/taud^2)*y(4);
% 
% %% Column B
% %Pyramidal
% dy(5)=y(13);
% dy(13)=(A/taue)*S7-(2/taue)*y(13)-(1/taue^2)*y(5);
% %Excitatory
% dy(6)=y(14)-y(15)+y(12);
% dy(14)=(A/taue)*(II+c2*S8)-(2/taue)*y(14)-(1/taue^2)*(y(6)+y(7)-y(4));
% %Inhibitory
% dy(7)=y(15);
% dy(15)=(B/taui)*c4*S9-(2/taui)*y(15)-(1/taui^2)*y(7);
% %Second column
% dy(8)=y(16);
% dy(16)=(A/taud)*KK*S10-(2/taud)*y(16)-(1/taud^2)*y(8);

% Sigmoid function
S1 = non_linear_sigmoid(y(5) - y(3), r, v0);% , sigma_sq_i); %(y(2)),v0,e0,r
S2 = non_linear_sigmoid(y(1), r, v0); % S((c1*y(1)),v0,e0,r);

% Equations
dy(1) = y(2);
% dy(2) = - 2*decay_e*y(2) - 1*decay_e^2*y(1) + alpha_i*S1;
dy(2) = - 2*decay_e*y(2) - 1*decay_e^2*y(1) + y(6)*S1;
dy(3) = y(4);
% dy(4) = - 2*decay_i*y(4) - 1*decay_i^2*y(3) + alpha_e*S2;
dy(4) = - 2*decay_i*y(4) - 1*decay_i^2*y(3) + y(7)*S2;
dy(5) = 0;
dy(6) = 0; % alpha_i
dy(7) = 0; % alpha_e

%% Column B
% %Pyramidal
% dy(5)=y(13);
% dy(13)=(A/taue)*S7-(2/taue)*y(13)-(1/taue^2)*y(5);
% %Excitatory
% dy(6)=y(14)-y(15);
% dy(14)=(A/taue)*(II+c2*S8+KK*y(4))-(2/taue)*y(14)-(1/taue^2)*(y(6)+y(7));
% %Inhibitory
% dy(7)=y(15);
% dy(15)=(B/taui)*c4*S9-(2/taui)*y(15)-(1/taui^2)*y(7);
% %Second column
% dy(8)=y(16);
% dy(16)=(A/taud)*S10-(2/taud)*y(16)-(1/taud^2)*y(8);
end

%% Sigmoid
function out=S(v,v0,e0,r)
    out=2*(e0)/(1+exp(r*(v0-v)));
end

%% Gaussian noise
function noise = p(t)
sd=0;
mean=220;
noise=mean+sd*randn(1,1);
end

%% Parameters
% y1=x1
% y2=x2
% y3=x3
% y4=v1
% y5=v2
% y6=v3