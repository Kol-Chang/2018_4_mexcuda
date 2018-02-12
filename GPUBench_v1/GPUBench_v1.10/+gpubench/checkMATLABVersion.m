function checkMATLABVersion()
%checkMATLABVersion  error if the MATLAB version is not suitable

%   Copyright 2011-2017 The MathWorks, Inc.

v = ver('MATLAB');
v = regexp( v.Version, '(?<major>\d+)\.(?<minor>\d+)', 'names' );
v.major = str2double( v.major );
v.minor = str2double( v.minor );
if (v.major < 8) || ((v.major==8) && (v.minor<2))
    error( 'GPUBench:MATLABVersion', 'GPUBench requires MATLAB version 8.2 (R2013b) or higher' );
end
