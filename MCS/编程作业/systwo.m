function [sys]=systwo(w,j)
n=w*w;
d=[1 2*w*j w*w];
sys=tf(n,d)
end