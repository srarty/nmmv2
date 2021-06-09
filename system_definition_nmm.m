% Coordinates
x1, x2, x3, x4, x5, x6, x7

% Parameters
r, v0, alpha_ie, alpha_ei, c1, c2, e_0, decay_e, decay_i, dt

% Sigmoid function
S1 = non_linear_sigmoid(y(5) - y(3), r, v0);
S2 = non_linear_sigmoid(y(1), r, v0);

% Equations
dy(1) = y(2);
dy(2) = - 2*decay_e*y(2) - 1*decay_e^2*y(1) + y(6)*S1;
dy(3) = y(4);
dy(4) = - 2*decay_i*y(4) - 1*decay_i^2*y(3) + y(7)*S2;
dy(5) = 0;
dy(6) = 0; % alpha_i = 162500
dy(7) = 0; % alpha_e = 440000