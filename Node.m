classdef Node
    properties
        Nodenums_of_Layers;
        Layer_index;
        Node_index;
        LearningRate;
        Weight;%n*1 matrix
        In;
        Out;
        Stashed_Error;
        Stashed_Weight;
    end
    methods
        function obj = Node(outdimension,LearningRate,Layer_index, Node_index,Nodenums_of_Layers)
            obj.Nodenums_of_Layers = Nodenums_of_Layers;
            obj.Layer_index = Layer_index;
            obj.Node_index = Node_index;
            obj.LearningRate = LearningRate;
            obj.Weight = rand(1,outdimension);
            obj.Stashed_Weight = obj.Weight;
            obj.In = 0;
            obj.Out = obj.activation(obj.In);
            obj.Stashed_Error = 0;
        end
        function out = activation(obj,in)%sigmoid
            out = 1/(1+exp(-in));
        end
        function obj = update(obj,in)
            % update weight
            obj.Weight = obj.Stashed_Weight;
            obj.In = in;
            obj.Out = activation(obj.In);
        end
        function [obj,DP_memory] = backpropgation(obj,DP_memory,Nodes_f,Nodes_ff)
            for i=1:size(obj.Weight,1)
                indices = obj.Nodenums_of_Layers(obj.Layer_index+2);
                indices = 1:1:indices;
                [V,M] = arrayfun(@(x) DP_memory.getStash(obj.Layer_index+2,x),indices);
                if(~isempty(find(V==false, 1)))
                    fprintf("error in get stash\n");
                end
                DP_memory = DP_memory.addStash( obj.Layer_index+1,i,data);
                M2 = arrayfun(@(x,y,z) x*(y.Out*(1-y.Out))*(z),M,Nodes_ff,Nodes_f(i).Stashed_Weight);
                gradient = sum(M2)*(Nodes_f(i).Out*(1-Nodes_f(i).Out)*obj.Out);
                obj.Stashed_Weight(i) = obj.Stashed_Weight(i)- obj.LearningRate*gradient;
            end
        end
    end
    
end