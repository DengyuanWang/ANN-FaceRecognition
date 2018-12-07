classdef ANN
    properties
       Lambda
       InputLayer      % 1*1 matrix num of nodes in input layer
       HiddenLayers % n*1 matrix num of nodes in input layer
       OutputLayer   % 1*1 matrix num of nodes in input layer
       Layers;             % n*1 cell array , every element is a node array(first node is bias node)
       LearningRate  % double
    end
    methods
        function obj = ANN(InputLayer, HiddenLayers, OutputLayer,LearningRate)
            %% init all parameters
            obj.Lambda = 1;
            obj.LearningRate = LearningRate;
            obj.InputLayer = InputLayer;
            obj.HiddenLayers = HiddenLayers;
            obj.OutputLayer = OutputLayer;
            numofLayers = [InputLayer;HiddenLayers;OutputLayer];
            obj.Layers = repmat({1},size(numofLayers,1),1);
            %% create nodes and layers
            for i=1:size(numofLayers,1)
                numofNodes = numofLayers(i);
                tmp = [];
                for j=1:numofNodes+1%add a bias node
                    if(i==size(numofLayers,1)) %output layer
                        % Node(out dimension, layerindex,nodeindex)
                       t = Node(1,i, j);t.Weight = 1;t.Stashed_Weight=1;
                       tmp = [tmp t];
                    else%other layers
                        % Node(out dimension, layerindex,nodeindex)
                       tmp = [tmp Node(numofLayers(i+1),i, j)];
                    end
                end
                obj.Layers{i} = tmp;
            end
        end
        function obj = backpropagation(obj,X,Y)
             A = obj.predixt(X);
             Deltas = [];
             M_deltas = [];
             for i=size(obj.Layers ,1):-1:1%loop from output layer to input layer
                if(i==size(obj.Layers ,1))%output layer
                    Delta = (A-Y)';
                else%for other layers
                    LastOut = arrayfun(@(x) x.Out,obj.Layers{i});%1*n mat
                    Theta = cell2mat(arrayfun(@(x) x.Weight,obj.Layers{i},'UniformOutput' , false));%m*n mat
                    %save delta(i+1)*A(i)^T;
%                     LastOut(1)=[];%delete output of bias
%                     Theta(:,1) = [];
                    M_deltas =  Delta*LastOut;
                    % delta(i) = Theta(i)^T*delta(i+1)*Out(i).*(1-Out(i))
                    Delta = Theta'* Delta.* (LastOut .*(1-LastOut))';
                    Delta(1) = [];
                    Theta(:,1) = 0;%set theta for bias node to 0
                    D_delta = M_deltas/size(X,1)+obj.Lambda *Theta;
                    D_delta = D_delta*obj.LearningRate;
                    D_delta = mat2cell(D_delta,size(D_delta,1),ones(1,size(D_delta,2)));
                    obj.Layers{i} = arrayfun(@(x,in) x.subWeight(in),obj.Layers{i}, D_delta);
                end
             end
        end
        function Y = predixt(obj,X)
            X = [0 X];% feed 0 in bias node
            for i=1:size(obj.Layers ,1)%loop from input layer to output layer
                if(i==1)%input layer
                    obj.Layers{i} = arrayfun(@(x,in) x.update(in),obj.Layers{i}, X);
                else%for other layers
                    LastOut = arrayfun(@(x) x.Out,obj.Layers{i-1});%1*n mat
                    Theta = cell2mat(arrayfun(@(x) x.Weight,obj.Layers{i-1},'UniformOutput' , false));%m*n mat
                    In = LastOut*Theta';%1*m mat
                    In = [0 In];% feed 0 in bias node
                    obj.Layers{i} = arrayfun(@(x,in) x.update(in),obj.Layers{i},In);
                end
            end
             LastOut = arrayfun(@(x) x.Out,obj.Layers{size(obj.Layers ,1)});%1*n mat
             Y = LastOut(2:end);%delete the bias node in output layer
        end
    end
end