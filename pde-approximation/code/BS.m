%-----------------------------------------------
%- Black & Scholes formulae for European put
%-----------------------------------------------

%- Black-Scholes formulae for the value of Vanilla European put at time t and stock price s.
function P=BS(t,s)
global K r sigma
if t==0 
    P=max(K-s,0);
else
    P=ones(size(s))*K*exp(-r*t); 
    i=find(s>0); 
    tau=sigma^2*t;
    dm=(log(s(i) /K) + r*t - 0.5*tau) / sqrt(tau);
    dp=(log(s(i) /K) + r*t + 0.5*tau) / sqrt(tau);
    P(i)=K*exp(-r*t)*(Normal(-dm)) - s(i).*(Normal(-dp));
end

function y=Normal(x)
  %y=cdf('normal',x,0,1);
  y=0.5*erf(x/sqrt(2))+0.5;
