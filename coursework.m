%% Generate 2 classes
close all, clear all, clc, format compact

C2 = [2 1 ; 1 2];
m2 = [0; 3];

C1 = [1 0 ; 0 1];
m1 = [2; 1];

sampleSize = 100;

% use matlab function for Multivariate normal random numbers samples
class1 = mvnrnd(m1, C1, sampleSize);
class2 = mvnrnd(m2, C2, sampleSize);

figure(1)
plot(class1(:,1),class1(:,2),'r.');
title('100 samples from each class');
hold on
plot(class2(:,1),class2(:,2),'mx');
hold off

%% Posterior Probability

% use matlab function to Construct Gaussian mixture distribution
dis = gmdistribution([m1'; m2'], cat(3, C1, C2));

% arbitrary axis
X(:,1) = -5:0.1:5; 
Y(:,1) = -5:0.1:5; 
P2 = [];
for i = 1:size(Y,1)	
    y = X(i)*ones(size(Y,1),1);
	P = posterior(dis, [ y Y(:,1)]);
	P2 = [P2 P(:,1)];
end
figure(2)
surf(X, Y, P2); xlabel('X'); ylabel('Y'); zlabel('Posterior Probability');
alpha(.5);
shading interp;
colormap cool;
hold on

%% Bayes’ optimal boundary

% make a 3D array to elevate sample data to 0.5 prob
classZ = zeros([100 3]);
classZ(:,3) = 0.5;
plot3(class1(:,1),class1(:,2),classZ(:,3),'r.');
title('100 samples from each class');
plot3(class2(:,1),class2(:,2),classZ(:,3),'mx');
contour3(X, Y, P2, [0.5 0.5], 'b'); % draw a line on contour at 0.5 prob

view(3)
title('Bayes’ optimal boundary');
hold on

%% contour line

alpha(0);
view(2)
%hold off

%% Neural Networks

figure(3)
%alpha(1);
% conbine samples from both classes for input
allClass = [class1' class2'];

target = [ones([1 100]) (zeros([1 100]))];

net = feedforwardnet(1); % 20 hidden layers feedforward network

net = train(net, allClass, target);
view(net);
%% prob
% generate a grid
[span1,span2] = meshgrid(X(:,1),Y(:,1));
nn = [span1(:) span2(:)]';

% simulate neural network on a grid
output = net(nn);

% plot classification regions
figure(4)

surf(span1, span2, reshape(output,length(X(:,1)),length(Y(:,1))));
shading interp;
alpha(.5);
%colormap cool
view(3) % view in 3D
%% prob with samples
%alpha(0);
hold on
contour3(span1, span2, reshape(output,length(X(:,1)),length(Y(:,1))), [0.5 0.5], 'r');

plot3(class1(:,1),class1(:,2),classZ(:,3),'r.');
plot3(class2(:,1),class2(:,2),classZ(:,3),'mx');
view(3)