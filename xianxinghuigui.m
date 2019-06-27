clear
clc
% create sigmal
m=500;
xi=0:1/m:1;
n=size(xi,2);
yi=1+3.*xi+xi.^2+0.1*randn(n,1)';


% least-square method,y=a0+a1*x
% [a0,a1]
A=[1,sum(xi)/n;1,sum(xi.^2)/sum(xi)];
b=[sum(yi);sum(xi*yi')/sum(xi)];
theta=inv(A)*b;
a1=(n*sum(xi*yi')-sum(xi)*sum(yi))/(n*sum(xi*xi')-sum(xi)*sum(yi));
a0=sum(yi)/n-a1*sum(xi)/n;
A=[ones(n,1),xi'];
% theta=A\yi';
theta=pinv(A)*yi';
a0=theta(1);
a1=theta(2);
hold on
x1=xi;
y1=a0+a1*x1;


% machine learning, gradient descent
clear a0 a1 A m
xn=xi';
% yn=a0+a1.*xn;
ex=ones(n,1);
xn=[ex,xn];
a0=[-5:25/(n-1):20];
a1=[-3:8/(n-1):5];
for i=1:n
    A(1,(i-1)*n+1:i*n)=a0(i);
    A(2,(i-1)*n+1:i*n)=a1;
end
yn=xn*A;
% half of the square
% for i=1:n
%     yn(:,i)=yn(:,i)-yi';
% end
J=1/2/n.*sum((yn-yi').^2);
for j=1:n
J0(j,:)=J((j-1)*n+1:j*n);
end
figure(1)
contour3(a0,a1,J0,150)
title('Gradient Descent')
M=2*n;
a01=interpft(a0,M);
a11=interpft(a1,M);
[X,Y]=meshgrid(a0,a1);
[Xi,Yi]=meshgrid(a01,a11);
Z=interp2(X,Y,J0,Xi,Yi,'cubic');
contour(Xi,Yi,Z,150)
% Gradient Descent
alpha=0.04;
% Grad
sigma0=20;
sigma1=5;
psigma0=sum(yn-yi',1)/n;
psigma1=xn(:,2)'*(yn-yi')/n;
hold on
i=1;
sigma00=sigma0+1;
while norm(sigma0-sigma00)>0.0001&i<20000
    sigma00=sigma0;
    sigma0=sigma0-alpha*psigma0(end);
    sigma1=sigma1-alpha*psigma1(end);
    [x0,y0]=min(abs(a0(:)-sigma0));
    [x01,y01]=min(abs(a1(:)-sigma1));
    psigma0=sum(yn(:,(y0-1)*n+1+y01)-yi',1)/n;
    psigma1=xn(:,2)'*(yn(:,(y0-1)*n+1+y01)-yi')/n;
    plot(sigma0,sigma1,'r*')
    i=i+1;
end
hold off
xm=xi;
ym=sigma0+sigma1*xm;

pause(3);
figure(2)
hold on
plot(xi,yi,'r*');
plot(x1,y1,'k-');
plot(xm,ym,'y-');
legend('Random-Signal','Least-Square','Grad-Descent')
hold off
grid on