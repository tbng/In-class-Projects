function ploot(t,s,P)
global K r sigma
global Xmin Xmax Ymin Ymax
global x_min x_max

% Pex=BS(t,s);

figure(1);
clf;
axis =[Xmin,Xmax,Ymin,Ymax];

%- copying here the ul, ur, u0 functions
global ul ur u0

sgraph=[x_min;s;x_max];
Pgraph=  [ul(t);P;  ur(t)];
% Pexgraph=[ul(t);Pex;ur(t)]; 


% plot(sgraph,Pexgraph,'black.-'); hold on;
% plot(sgraph,'black.-'); hold on;
plot(sgraph,Pgraph,'blue.-');
%plot(sgraph,100*(Pgraph-Pexgraph),'red.-');
titre=strcat('t=',num2str(t)); title(titre);
xlabel('x');
ylabel('Price');
grid;

