function [p]=systwo(t)
K=1;t1=2;t2=0.5;
sys1=tf([-K*t1*t 2*t1*K-t*K 2*K],[t2*t t+2*t2 2]);
sys2=tf(-10,[1 10]);
sys3=tf([-1 -6],[1 3 6 0]);
sys_tmp=series(sys1,sys3);
sys=series(sys_tmp,sys2);
sys=feedback(sys,1);
q=[sys.den{1}];
p=roots(q);
end