% using MC simulation to calculate the price of Asian option
%------------------------
%- FINANCIAL PARAMETERS
%------------------------

global s0 r sigma T K1 K2
s0= 100; T= 1; r= 0.03; sigma= 0.2;
K1= 100; K2= 110;

%------------------------
%- NUMERICAL DATA
%------------------------

M= 'Enter the number of timesteps M: ';
M= input(M);
%M= 250;
N= 'Enter the number of simulations N: ';
N= input(N);
%N= 100000;
%-------------------------
%- IMPLEMENTATION
%-------------------------
dt = T/M; % timestep
% declare the drift and volatilty following the settings
drift = exp(dt * (r - 0.5 * sigma ^2 ));
vol = sqrt(dt) * sigma;
spot_price= ones(M,1)*s0;
asian_price = 0;
payoff1 = 0; % payoff for Asian fixed strike call
varpayoff1 = 0;
payoff2 = 0; % payoff for Asian binary option
varpayoff2 = 0;
%
tic();
for i=1:N
    for j=2:M
        spot_price(j)= spot_price(j-1) * drift * exp(vol*randn()); 
        
    end
    asian_price = mean(spot_price);
    payoff1 = payoff1 +  max(asian_price-K1,0);
    varpayoff1 = varpayoff1 + (max(asian_price-K1,0))^2;
    if K1 < asian_price && asian_price < K2
        payoff2 = payoff2 + 1;
        varpayoff2 = varpayoff2 + 1;
    end
end

payoff1 = 1/N*exp(-r*T)*payoff1;
varpayoff1 = N/(N-1)* (1/N*exp(-2*r*T)*varpayoff1 - (payoff1)^2);

payoff2 = 1/N*exp(-r*T)*payoff2;
varpayoff2 = N/(N-1)* (1/N*exp(-2*r*T)*varpayoff2 - (payoff2)^2);

%confidence interval (for 95%)
lbound1 = payoff1 - 1.96*sqrt(varpayoff1)/sqrt(N);
ubound1 = payoff1 + 1.96*sqrt(varpayoff1)/sqrt(N);
lbound2 = payoff2 - 1.96*sqrt(varpayoff2)/sqrt(N);
ubound2 = payoff2 + 1.96*sqrt(varpayoff2)/sqrt(N);
total_time = toc();
%------------------------
%- PRINTING RESULTS
%------------------------

fprintf('======================\n')
fprintf('Monte Carlo simulation\n\n')
%fprintf('======================\n')
fprintf('Parameters: K1= %3i, K2= %3i, s0= %3i, r= %3.2f, sigma= %3.2f\n', K1, ...
        K2, s0, r, sigma)
fprintf('Number of time steps M= %5i, number of simulations N= %5i\n', M, N)
fprintf('Total time running (sec) = %5.2f\n\n', total_time)
%fprintf('======================\n')
fprintf('Price1= %5.6f, varPrice1= %5.6f\n', payoff1, varpayoff1)
fprintf('95 percent CI = [%5.6f, %5.6f]\n\n', lbound1, ubound1)

fprintf('Price2= %5.6f, varPrice2= %5.6f\n', payoff2, varpayoff2)
fprintf('95 percent CI = [%5.6f, %5.6f]\n', lbound2, ubound2)
fprintf('======================\n')