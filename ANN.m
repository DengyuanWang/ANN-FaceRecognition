classdef ANN
    properties
       Lambda
       InputLayer      % 1*1 matrix num of nodes in input layer
       HiddenLayers % n*1 matrix num of nodes in input layer
       OutputLayer   % 1*1 matrix num of nodes in input layer
       Layers;             % n*1 cell array , every element is a node array(first node is bias node)
       LearningRate  % double
       Inputs;
       Outputs;
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
            tmp = arrayfun(@(i) ...
                           repmat(Node(numofLayers(i+1),i),1,numofLayers(i)+1) ...
                            ,1:size(numofLayers,1)-1,'UniformOutput' , false);
            i = size(numofLayers,1);
            tmp = [tmp {arrayfun(@(x) Node(1,i,'outputlayer'),1:numofLayers(i)+1)}];
            obj.Layers = tmp';
            
        end
        function obj = backpropagation(obj,X,Y)
             A = obj.predixt(X);
             for i=size(obj.Layers ,1):-1:1%loop from output layer to input layer
                if(i==size(obj.Layers ,1))%output layer
                    Delta = (A-Y)';
                else%for other layers
                    [Out,Theta] = obj.get_info(i);
                    Theta = Theta';
                    M_deltas =  Delta*Out;
                    % delta(i) = Theta(i)^T*delta(i+1)*Out(i).*(1-Out(i))
                    Delta = Theta'* Delta.* (Out .*(1-Out))';
                    Delta(1) = [];
                    Theta(:,1) = 0;%set theta for bias node to 0
                    D_delta = M_deltas/size(X,1)+obj.Lambda *Theta;
                    D_delta = D_delta*obj.LearningRate;
                    D_delta = mat2cell(D_delta,size(D_delta,1),ones(1,size(D_delta,2)));
                    tmp = obj.Layers{i};
                    parfor j=1:size(obj.Layers{i},2)
                      tmp(j) = tmp(j).subWeight(D_delta(j));
                    end
                    obj.Layers{i} = tmp;
%                     obj.Layers{i} = arrayfun(@(x,in) x.subWeight(in),obj.Layers{i}, D_delta);
                end
             end
        end
        function obj = set_theta(obj,Thetas)
            for i=1:size(obj.Layers ,1)%loop from input layer to output layer
                theta_now = Thetas{i};
                layers_now = obj.Layers{i};
                parfor j=1:size(obj.Layers(i),2)
                    layers_now(j).Stashed_Weight = layers_now(j).Stashed_Weight-theta_now(:,j);
                    layers_now(j).Weight = layers_now(j).Weight;
                end
                obj.Layers{i} = layers_now;
            end
        end
        function Thetas = get_deltatheta(obj)
            Thetas = repmat({0},size(obj.Layers ,1),1);
            for i=1:size(obj.Layers ,1)%loop from input layer to output layer
                layers_now = obj.Layers{i};
                tmp = zeros(size(layers_now(1).Weight,1),size(layers_now,2));%weight_dim*sample_dim
                parfor j=1:size(layers_now,2)
                    tmp(:,j) = layers_now(j).Weight-layers_now(j).Stashed_Weight;
                end
                Thetas{i} = tmp;
            end
        end
        function Y = predixt(obj,X)
            In = [0 X];% feed 0 in bias node
            for i=1:size(obj.Layers ,1)%loop from input layer to output layer
                if(i~=1)%for other layers
                    [Out,Theta] = obj.get_info(i-1);
                    Out(1) = 1;%bias node 
                    In = [0 Out*Theta];%1*(m+1) mat % feed 0 in bias node
                end
                tmp = obj.Layers{i};
                parfor j=1:size(obj.Layers{i},2)
                  tmp(j) = tmp(j).update(In(j));
                end
                obj.Layers{i} = tmp;
            end
            [Out,~] = get_info(obj,size(obj.Layers ,1));
             Y = Out(2:end);%delete the bias node in output layer
        end
        function [Out,Theta] = get_info(obj,layer_index)
            tmp = obj.Layers{layer_index};
            Out = zeros(size(tmp));Theta = zeros(size(tmp,2),size(tmp(1).Weight,1));
            parfor j=1:size(obj.Layers{layer_index},2)
              Out(j) = tmp(j).Out;
              Theta(j,:) = tmp(j).Weight;
            end
        end
    end
end