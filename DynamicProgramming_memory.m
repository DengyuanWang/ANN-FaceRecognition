classdef DynamicProgramming_memory
    properties
        Final_Error;
        Stash;%n*1 matrix 
        HashMap;%container
    end
    methods
        function obj = DynamicProgramming_memory(Error)
            obj.Final_Error = Error;
            obj.HashMap = containers.Map;
        end
        function obj = addStash(obj,layer_index,node_index,data)
           %data = ?Final_Error/??Node(layer_index,node_index).Out?
            str = layer_index+":"+node_index;
            obj.HashMap(str) = size(obj.Stash,1)+1;
            obj.Stash = [obj.Stash;data];
        end
        function [valid_tag,data] = getStash(obj,layer_index,node_index)
            str = layer_index+":"+node_index;
            hash_index =  obj.HashMap(str);
            if(hash_index<=size(obj.Stash,1)&&hash_index>0)
                data = obj.Stash(hash_index);
                valid_tag = true;
            else
                valid_tag = false;
                data = nan;
            end
        end
    end
end