function [out_struct] = grassgraph_assoc_test(in_struct)
    %tic
    X = in_struct.X;
    Y = in_struct.Y;
    %p = grassGraphsParams_Clean;                        % GrassGraphs parameters.
    p = in_struct.p;
    %toc

    %tic
    [corr, rA] = grassGraphsMatching(X, Y, 0, p);
    %toc
    %size(X)
    %size(Y)
    %size(corr.X)
    %size(corr.Y)
    
    %disp("rA is: ")
    %disp(rA)
    %disp("corr.X is: ")
    %disp(corr.X)
    %disp("corr.Y is: ")
    %disp(corr.Y)
    
    %tic
    %disp(append('size(corr.X): ', mat2str(size(corr.X))))
    Xh = [corr.X ones(size(corr.X,1),1)];
    %disp(append('size(Xh): ', mat2str(size(Xh))))
    XrA = Xh*rA;                                        % Form the recovered shape.
    %disp(append('size(rA): ', mat2str(size(rA))))
    %disp(append('size(XrA): ', mat2str(size(XrA))))
    XrA = XrA(:,1:3);                                   % Remove the extra homogeneous dimension.
    %disp(append('size(XrA): ', mat2str(size(XrA))))
    %toc
    
    %disp("Xh is: ")
    %disp(Xh)
    %disp("XrA is: ")
    %disp(XrA)
    
    %tic
    out_struct.rA = rA;
    out_struct.diff = (corr.Y - XrA);
    %out_struct.p = p
    out_struct.num_matches = size(corr.X,1);
    %toc
    
    % added subsiquently to extend output contents
    out_struct.corrX = corr.X;
    out_struct.corrY = corr.Y;
    out_struct.XrA = XrA;
    
    
end

