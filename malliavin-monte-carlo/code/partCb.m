% calculating delta using finite difference method and MC estimation for
% price function
%------------------------
%- FINANCIAL PARAMETERS
%------------------------

global s0 r sigma T K1 K2
s0 = 100; T= 1; r= 0.03; sigma= 0.2;
K1= 100; K2= 110;

%------------------------
%- NUMERICAL DATA
%------------------------

M= 'Enter the number of timesteps M: ';
M= input(M);
%M= 50;
N= 'Enter the number of simulations N: ';
N= input(N);
%N= 100000;
%-------------------------
%- IMPLEMENTATION
%-------------------------
dt = T/M; % timestep
% declare the drift and volatilty following the settings
epsi= 'Enter the value of epsilon: ';
epsi= input(epsi);
%epsi= 0.1;
drift = exp(dt * (r - 0.5 * sigma ^2 ));
vol = sqrt(dt) * sigma;

%used for calculating P(x-epsilon) and P(x+epsilon)
spot_price_low= ones(M,1)*(s0-epsi);
spot_price_high= ones(M,1)*(s0+epsi);

asian_price_low = 0;
asian_price_high = 0;

delta1 = 0; % delta for Asian fixed strike call
vardelta1 = 0;

payoff2 = 0;
delta2 = 0; % delta for Asian binary option
vardelta2 = 0;

% MC loop
tic();
for i=1:N
    A = randn(M,1); %generate Guassian vector - using same random sample as
                    %explained in the lecture note
    for j=2:M
        spot_price_low(j)= spot_price_low(j-1) * drift * exp(vol*A(j));
        spot_price_high(j)= spot_price_high(j-1) * drift * exp(vol*A(j));        
    end
    asian_price_low = mean(spot_price_low);
    asian_price_high = mean(spot_price_high);
    delta1 = delta1 + (max(asian_price_high-K1,0) - max(asian_price_low-K1,0))/(2*epsi);
    vardelta1 = vardelta1 + 1/(4*epsi^2)*(max(asian_price_high-K1,0) - max(asian_price_low-K1,0))^2;
    if K1 < asian_price_low && asian_price_low < K2
        payoff2_low = 1;
    else
        payoff2_low = 0;
    end
    if K1 < asian_price_high && asian_price_high < K2
        payoff2_high = 1;
    else
        payoff2_high = 0;
    end
    delta2 = delta2 + (payoff2_high - payoff2_low)/(2*epsi);
    vardelta2 = vardelta2 + 1/(4*epsi^2)*(payoff2_high - payoff2_low)^2;
end

delta1 = 1/N*exp(-r*T)*delta1;
delta2 = 1/N*exp(-r*T)*delta2;
vardelta1 = N/(N-1) * (1/N*exp(-2*r*T)*vardelta1 - (delta1)^2);
vardelta2 = N/(N-1) * (1/N*exp(-2*r*T)*vardelta2 - (delta2)^2);

%confidence interval (for 95%)
lbound_delta1 = delta1 - 1.96*sqrt(vardelta1)/sqrt(N);
ubound_delta1 = delta1 + 1.96*sqrt(vardelta1)/sqrt(N);
lbound_delta2 = delta2 - 1.96*sqrt(vardelta2)/sqrt(N);
ubound_delta2 = delta2 + 1.96*sqrt(vardelta2)/sqrt(N);

total_time = toc();

%------------------------
%- PRINTING RESULTS
%------------------------

fprintf('======================\n')
fprintf('Monte Carlo simulation\n')
fprintf('======================\n')
fprintf('Parameters: K1= %3i, K2= %3i, s0= %3i, r= %3.2f, sigma= %3.2f\n', K1, ...
        K2, s0, r, sigma)
fprintf('Number of time steps M= %5i, number of simulations N= %5i\n', M, N)
fprintf('Total time running (sec) = %5.2f\n', total_time)
fprintf('======================\n')
fprintf('With epsilon = %5.6f\n\n', epsi)
fprintf('delta1= %5.6f, vardelta1= %5.6f\n', delta1, vardelta1)
fprintf('95 percent CI = [%5.6f, %5.6f]\n\n', lbound_delta1, ubound_delta1)

fprintf('delta2= %5.6f, vardelta2= %5.6f\n', delta2, vardelta2)
fprintf('95 percent CI = [%5.6f, %5.6f]\n', lbound_delta2, ubound_delta2)
fprintf('======================\n')