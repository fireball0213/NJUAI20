k=10.8E8;a=1;b=8;J=10.8E8;
num1=[k k*a];
den1=[1 b];
num2=1;
den2=[J 0 0];
[numc,denc]=series(num1,den1,num2,den2);
%Q1
sys=tf (numc, denc)
sys1=feedback(sys,1)%别忘记加反馈
%Q2
t=[0:0.1:100];
sys2=10*sys1;%输入10幅值的阶跃，乘10即可
y=step(sys2,t);
% plot(t,y),grid
%Q3:80%
J=0.8*J
den2=[J 0 0];
[numc,denc]=series(num1,den1,num2,den2);
sys=tf (numc, denc);
sys1=feedback(sys,1);
sys2=10*sys1;
y1=step(sys2,t);
%Q3:50%
J=0.5*J
den2=[J 0 0];
[numc,denc]=series(num1,den1,num2,den2);
sys=tf (numc, denc);
sys1=feedback(sys,1);
sys2=10*sys1;
y2=step(sys2,t);
plot(t,y,t,y1,'--',t,y2,'.'),grid