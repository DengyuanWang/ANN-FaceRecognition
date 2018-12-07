classdef ANN
    properties
       InputLayer
       HiddenLayers
       OutputLayer
       Layers;
       LearningRate
    end
    methods
        function obj = ANN(InputLayer, HiddenLayers, OutputLayer,LearningRate)
            obj.LearningRate = LearningRate;
            obj.InputLayer = InputLayer;
            obj.HiddenLayers = HiddenLayers;
            obj.OutputLayer = OutputLayer;
            numofLayers = [InputLayer;HiddenLayers;OutputLayer];
            obj.Layers = repmat({1},size(numofLayers,1),1);
            
            for i=1:size(numofLayers,1)
                numofNodes = numofLayers(i);
                tmp = [];
                for j=1:numofNodes
                    if(i==size(numofLayers,1)) 
                       tmp = [tmp Node(1,obj.LearningRate,i, j,numofLayers)];
                    else
                       tmp = [tmp Node(numofLayers(i+1),obj.LearningRate,i, j,numofLayers)];
                    end
                end
                obj.Layers{i} = tmp;
            end
        end
    end
end