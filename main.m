function net1 = main()
    InputLayer = 10;
    HiddenLayers = [100 100 100]';
    OutputLayer = [62];
    LearningRate = 0.01;
    net1 = ANN(InputLayer, HiddenLayers, OutputLayer,LearningRate);
end