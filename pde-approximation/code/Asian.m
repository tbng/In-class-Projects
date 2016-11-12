%- Finite Difference - Asian Fixed Strike
clear

%------------------------
%- FINANCIAL PARAMETERS
%- Following method of Roger & Shi (95)
%------------------------
% X = (K - t*A/T)*1/S
global  K r sigma T
s0= 100; T= 1;
K= 'Enter the value of K: ';
K= input(K);
%K= 100;
sigma= 'Enter the value of sigma: ';
sigma= input(sigma);
%sigma= 0.3;
r= 'Enter the value of r: ';
r= input(r);
%r= 0.15;


%------------------------
%- NUMERICAL DATA
%------------------------
%I= 'Enter the value of I: ';
%I= input(I);

N= 'Enter the value of N: ';
N= input(N);
I= 1.5*N;
% I follows the setting in Dubois & Lelievre's paper
global x_min x_max
x_max = 2; x_min = 0;


%- IC : Initial Condition   = function  u0
%- BD : Boundary Conditions = functions ul, ur
%- ==> COMPLETE definition of functions u0, ul, ur (inline definitions: see below)
global ul ur
u0= @(x) max(-x,0);		%- Initial values (payoff function)
ul= @(t) -x_min*exp(-r*t) + (1 - exp(-r*t))/(r*T);	%- ul= left  value, at x_min
ur= @(t) 0;			%- ur= right value, at x_max

%-------------------------
%- Specifying the Scheme and Difference method
%-------------------------
scheme='CN'; 		%- 'EE' or 'EI' or 'CN' 
difference_type='CENTER'; 	%- 'CENTER', 'RIGHT', 'LEFT' 

%----------- Parameters for the graphics:
global Xmin Xmax 
Xmin=x_min; Xmax=x_max;
err_scale=0; %- Echelle pour le graphe d'erreur.
deltan=N/10; %- Eventuellement, Affichage uniquement tous les deltan pas.

%- Printing some data:
fprintf('K=%5i, s0=%5i\n', K, s0)
fprintf('sigma=%5.2f, r=%5.2f, x_max=%5.2f, x_min=%5.2f\n', sigma, r, x_max,x_min);
fprintf('Asset step I= %5i, Time step N=%5i\n', I, N);
fprintf('difference_type : %s\n', difference_type);
fprintf('scheme: %s\n', scheme);

%-----------------------------------------------
%- Black - Scholes formula
%-----------------------------------------------
%- Black & Scholes formulae for vanila European option: BS.m
%- function y=BS(t,s,K,r,sigma)


%--------------------
%- MAILLAGE / MESH
%--------------------
%- FILL : dt, h, s (time step, mesh step, mesh) 

dt=T/N; 		%- time step
h=(x_max-x_min)/(I+1); 	%- mesh step
x=x_min+(1:I)'*h; 	%- mesh : column vector of size I,
                        %- containing the mesh values x_i = x_min + i*h

%- CFL (Courant-Friederichs-Lewy) COEFFICIENT 
%COMPLETE
cfl=dt/h^2 * (sigma*x_max)^2;
fprintf('CFL : %5.3f\n',cfl); 


%--------------------------
%- Initializations:  matrix A, function q(t)
%--------------------------

switch difference_type

case 'CENTER';  %- CENTRAL DIFFERENCES

  %- FILL IN / Matrix A and function q(t)
  A=zeros(I,I);
  % FILL the values of A(i,i), A(i,i-1), A(i,i+1)
  alpha=sigma^2/2 * x.^2 /h^2;
  bet=(1/T + r*x)/h;
  for i=1:I;   A(i,i) = 2*alpha(i); end;
  for i=2:I;   A(i,i-1) = -alpha(i) - bet(i)/2; end;
  for i=1:I-1; A(i,i+1) = -alpha(i) + bet(i)/2; end;

  % FILL IN
  q = @(t) [(-alpha(1) - bet(1)/2)* ul(t);  zeros(I-2,1);  (-alpha(end) + bet(end)/2)* ur(t)];

case 'RIGHT';	%- FORWARD DIFFERENCES

  A=zeros(I,I);
  alpha=sigma^2/2 * x.^2 /h^2;
  bet=(1/T + r*x)/h;
  for i=1:I;   A(i,i) = 2*alpha(i) - bet(i); end;
  for i=2:I;   A(i,i-1) = -alpha(i) ; end;
  for i=1:I-1; A(i,i+1) = -alpha(i) + bet(i); end;

  q = @(t) [(-alpha(1))* ul(t);  zeros(I-2,1);  (-alpha(end) + bet(end))* ur(t)];

case 'LEFT';	%- BACKWARD DIFFERENCES

  A=zeros(I,I);
  alpha=sigma^2/2 * x.^2 /h^2;
  bet=(1/T + r*x)/h;
  for i=1:I;   A(i,i) = 2*alpha(i) + bet(i); end;
  for i=2:I;   A(i,i-1) = -alpha(i) - bet(i) ; end;
  for i=1:I-1; A(i,i+1) = -alpha(i) ; end;

  q = @(t) [(-alpha(1) - bet(1))* ul(t);  zeros(I-2,1);  (-alpha(end))* ur(t)];

otherwise 

  fprintf('DIFFERENCE_TYPE not specified !'); abort

end

%--------------------
%- Initialiser P et graphique
%--------------------
P=u0(x);
ploot(0,x,P);
fprintf('waiting for ''Enter''\n'); input('');

 
%--------------------
%- BOUCLE PRINCIPALE / MAIN LOOP
%--------------------
%- starting cputime counter
tic(); 

Id=eye(size(A));

for n=0:N-1

  t=n*dt;

  %- Scheme
  switch scheme 
  case 'EE'; % Euler Explicit
    % COMPLETER
    P =  (Id - dt*A)*P - dt*q(t);

  case 'EI'; % Euler Implicit 
    % Using linear solver in Matlab with backslash operator
    t1=t+dt; 
    P = (Id + dt*A)\(P-dt*q(t1)); 

  case 'CN'; % Crank Nicholson
    % Using linear solver in Matlab with backslash operator
    q0=q(t);
    q1=q(t+dt);
    P = (Id + dt/2*A) \ ( (Id - dt/2*A) * P - dt*(q0+q1)/2 );

  otherwise
    fprintf('SCHEME is not specified correctly'); abort;

  end

  if mod(n+1,deltan)==0; 	%- Printings at each deltan steps.

   %- Graphs:
   t1=(n+1)*dt; 
   ploot(t1,x,P); pause(1e-3);

   %- Error computations:
   % COMPLETER errLI
   %Pex=10.209774;		%- Reference value
   %errLI=norm(P-Pex,'inf');	%- calculate the 2-norm of errLI, with 'inf'
                                %- then errLI = max(abs(P-Pex))
   %fprintf('t=%5.2f; Err.Linf=%8.5f',t1,errLI);  
   %fprintf('\n');
   %input('');
  end

   
end

% calculate the price of Asian fixed strike call option at time 0
% using interpolation of degree 2

x_bar = K/s0;
z = floor((x_bar-x_min)/h);
lambda = (x(z+1)-x_bar)/h;
p_call = s0*(lambda*P(z)+(1-lambda)*P(z+1)); % price of the option at time 0
fprintf('Price = %5.6f\n', p_call);

t_total=toc();
fprintf('total time = %5.2f\n',t_total);
fprintf('program ended normaly\n');

