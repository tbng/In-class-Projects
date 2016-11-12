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
M= 50;
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

delta1 = 0; %delta for Asian fixed strike call
vardelta1 = 0;

delta2 = 0; %delta for Asian binary option
vardelta2 = 0;
%
tic();
for i=1:N
    A = randn(); % use this to generate B_t1 to calculate Pi;
    B_t1 = sqrt(dt) * A;
    spot_price(2)= spot_price(1) * drift * exp(vol*A);
    for j=3:M
        spot_price(j)= spot_price(j-1) * drift * exp(vol*randn());
    end
    asian_price = mean(spot_price);
    multiplier1 = max(asian_price-K1,0)*B_t1/(s0*sigma*dt);
    delta1 = delta1 + multiplier1;
    vardelta1 = vardelta1 + multiplier1^2;
    if K1 < asian_price && asian_price < K2
        multiplier2 = B_t1/(s0*sigma*dt);
    else
        multiplier2 = 0;
    end
    delta2 = delta2 + multiplier2;
    vardelta2 = vardelta2 + multiplier2^2;
end

delta1 = 1/N*exp(-r*T)*delta1;
vardelta1 = N/(N-1)* (1/N*exp(-2*r*T)*vardelta1 - (delta1)^2);

delta2 = 1/N*exp(-r*T)*delta2;
vardelta2 = N/(N-1)* (1/N*exp(-2*r*T)*vardelta2 - (delta2)^2);

%confidence interval (for 95%)
lbound1 = delta1 - 1.96*sqrt(vardelta1)/sqrt(N);
ubound1 = delta1 + 1.96*sqrt(vardelta1)/sqrt(N);
lbound2 = delta2 - 1.96*sqrt(vardelta2)/sqrt(N);
ubound2 = delta2 + 1.96*sqrt(vardelta2)/sqrt(N);
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
fprintf('delta1= %5.6f, vardelta1= %5.6f\n', delta1, vardelta1)
fprintf('95 percent CI = [%5.6f, %5.6f]\n\n', lbound1, ubound1)

fprintf('delta2= %5.6f, vardelta2= %5.6f\n', delta2, vardelta2)
fprintf('95 percent CI = [%5.6f, %5.6f]\n', lbound2, ubound2)
fprintf('======================\n')