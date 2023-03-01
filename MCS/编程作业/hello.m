
% 
% num=[10 60];
% den=conv([1  0 10],[1])
% sys=tf(num, den);
% % sys=feedback(sys,tf([1],[1 0]))
% rlocus(sys)
% % rlocfind(sys)
% p=[1 7 10 6]
% roots()
% G=tf([1],[0.5 1.5 1]);
% C=tf([0.82 0.9],[1 0]);
% 
% % sys = series (G, C)
% sys=tf([0.82 0.9],[0.5 1.5 1 0])
% sys_cl=feedback(sys,1)
% t=[0:0.01:20];
% sys1=plotmystep(1)
% sys2=plotmystep(1.5);
% sys3=plotmystep(2.85);
% [y1,t]=step(sys1,t);
% [y2,t]=step(sys2,t);
% [y3,t]=step(sys3,t);
% plot(t,y1,t,y2,t,y3),grid
G=tf([120 720],[1 0 10]);
t=[0:0.01:20];
[y1,t]=step(sys1,t);
plot(t,y1),grid
% sisotool(G)
