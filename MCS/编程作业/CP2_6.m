%Q1
num1=[1 0];den1=[1 1 2 2];
num2=[4 2];den2=[1 2 1];
sys1=tf(num1,den1);
sys2=tf(num2,den2);
sys_cl1=feedback(sys1,sys2,-1);
sys3=tf(1,[1 0 0]);
sys_cl2=feedback(sys3,50,1);
sys4=series(sys_cl1,sys_cl2);
sys5=tf([1 0 2],[1 0 0 14]);
sys_cl3=feedback(sys4,sys5,-1);
sys=series(4,sys_cl3)
sys=series(tf(4,1),sys_cl3)