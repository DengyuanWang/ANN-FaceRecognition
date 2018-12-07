classdef Node
    properties
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
        function obj = Node(outdimension,Layer_index, Node_index)
            obj.Layer_index = Layer_index;
            obj.Node_index = Node_index;
            obj.Weight = rand(outdimension,1);
            obj.Stashed_Weight = obj.Weight;
            obj.In = 0;
            obj.Out = activation(obj.In);
            obj.Stashed_Error = 0;
        end
        function obj = update(obj,in)
            % update weight
            obj.Weight = obj.Stashed_Weight;
            obj.In = in;
            if(obj.Node_index==1)%bias node
                obj.Out = 1;
            else
                obj.Out = activation(obj.In);
            end
        end
        function obj = subWeight(obj,delta_weights)
             obj.Stashed_Weight = obj.Stashed_Weight - delta_weights{1};
        end
    end
end
function out = activation(in)%sigmoid
    out = 1/(1+exp(-in));
end