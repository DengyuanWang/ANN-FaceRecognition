function Y = main()
%     InputLayer = 10;
%     HiddenLayers = [100 100 100]';
%     OutputLayer = [62];
% X and Y need to be 1*n1 and 1*n2
    X = [0 1];Y = [1 0];
    InputLayer = size(X,2);
    HiddenLayers = [20  20 20]';
    OutputLayer =  size(Y,2);
    LearningRate = 0.001;
    net1 = ANN(InputLayer, HiddenLayers, OutputLayer,LearningRate);
    Y = net1.predixt([0 1]);

    for i=1:1000
        X = [0 1];Y = [0 1];
        net1 = net1.backpropagation(X,Y);
    end
    Y = net1.predixt([0 1]);
end