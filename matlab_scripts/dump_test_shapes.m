function [out_struct] = dump_test_shapes(in_struct)
    shName = 'COSEG'; 
    load(shName);                   % A cell name shapeCell is loaded. Each cell is a shape.
    
    shNum = in_struct.shape_num;    % There are 50 shapes.
    X = shapeCell{shNum};
    
    out_struct.shape = X;
end

