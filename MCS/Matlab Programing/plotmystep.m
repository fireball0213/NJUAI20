function [sys]=plotmystep(K)
sys1=tf([2*K 2*K],[1 4 0]);
sys=0.6*feedback(sys1,tf([3],[1 2 5]),-1)
% sys=tf([0.6*[2 6 14 10]],[1 6 13 26 6])
end