%https://www.mathworks.com/help/distcomp/examples/improve-performance-of-element-wise-matlab-functions-on-the-gpu-using-arrayfun.html

hornerFcn = @horner;

data1  = rand( 4000, 'single' );
data2  = rand( 2000, 'double' );
gdata1 = gpuArray( data1 );
gdata2 = gpuArray( data2 );

gresult1 = arrayfun( hornerFcn, gdata1 );
gresult2 = arrayfun( hornerFcn, gdata2 );

comparesingle = max( max( abs( gresult1 - horner( data1 ) ) ) );
comparedouble = max( max( abs( gresult2 - horner( data2 ) ) ) );

fprintf( 'Maximum discrepancy for single precision: %g\n', comparesingle );
fprintf( 'Maximum discrepancy for double precision: %g\n', comparedouble );

% CPU execution
tic
hornerFcn( data1 );
tcpu = toc;

% GPU execution using only gpuArray objects
tgpuObject = gputimeit(@() hornerFcn(gdata1));

% GPU execution using gpuArray objects with arrayfun
tgpuArrayfun = gputimeit(@() arrayfun(hornerFcn, gdata1));

fprintf( 'tcpu : %f\n', tcpu);
fprintf( 'Speed-up achieved using gpuArray objects only: %g\n',...
    tcpu / tgpuObject );
fprintf( 'Speed-up achieved using gpuArray objects with arrayfun: %g\n',...
    tcpu / tgpuArrayfun );

function y = horner(x)
%HORNER - series expansion for exp(x) using Horner's rule
y = 1 + x.*(1 + x.*((1 + x.*((1 + ...
        x.*((1 + x.*((1 + x.*((1 + x.*((1 + ...
        x.*((1 + x./9)./8))./7))./6))./5))./4))./3))./2));
end